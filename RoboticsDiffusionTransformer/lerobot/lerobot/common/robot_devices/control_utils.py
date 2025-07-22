# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################

from PIL import Image as PImage
import cv2
import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import os
import cv2
import numpy as np
from PIL import Image as PImage
import pandas as pd

import rerun as rr
import torch
from deepdiff import DeepDiff
from termcolor import colored
import wandb

from lerobot.lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lerobot.common.datasets.utils import get_features_from_robot
from lerobot.lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.lerobot.common.robot_devices.robots.utils import Robot
from lerobot.lerobot.common.robot_devices.utils import busy_wait
from lerobot.lerobot.common.utils.utils import get_safe_torch_device, has_method

import time

def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)

@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True

def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            # Skip all observations that are not tensors (e.g. text)
            if not isinstance(observation[name], torch.Tensor):
                continue

            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action

def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events

def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_data,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_data=display_data,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )

def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    policy,
    fps,
    single_task,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
    )


import torch


def interp_action(current_robot_action, predicted_plan, max_deg_per_step=0.4):

    if predicted_plan.shape[0] == 0:
        return torch.empty_like(predicted_plan)

    max_rad_per_step = torch.deg2rad(torch.tensor(max_deg_per_step))
    first_planned_action = predicted_plan[0]

    delta = torch.abs(first_planned_action[:5] - current_robot_action[:5])

    max_delta = torch.max(delta)

    if max_delta <= max_rad_per_step:
        return predicted_plan
    else:
        num_interp_steps = torch.ceil(max_delta / max_rad_per_step)
        num_interp_steps = int(num_interp_steps.item())

        interpolated_steps = []
        for i in range(1, num_interp_steps + 1):
            weight = i / num_interp_steps
            step = torch.lerp(current_robot_action.unsqueeze(0), first_planned_action.unsqueeze(0), weight)
            interpolated_steps.append(step)

        final_plan = torch.cat(interpolated_steps + [predicted_plan[1:]], dim=0)
        return final_plan

def change2isaac(action):
    # #import pdb;pdb.set_trace()
    B1 = torch.tensor([1064.,863.,1070.,1080,450,2040])
    C1 = torch.tensor([990.,1093.,930.,910.,1483.,1210.])
    X1 = torch.tensor([1.5,1.5,1.5,1.5,2.7,1.5])
    D1 = torch.tensor([0,-0.15,0,0,0,-0.17])
    action['action'] = (torch.clamp(action['action'] - B1 - C1, min=-C1, max=C1) / C1) * X1  + D1
    return action

@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_data=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
    simoperate = False,
    rdt = False,
    rdt_policy = None,
    lang_emd = None,
    log_to_wandb=True,
    wandb_log_freq=10,
    q_sim = None
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")
    
    if simoperate and policy is not None:
        raise ValueError("When `simoperate` is True, `policy` should be None.")    

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    # Controls starts, if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    # --- Define normalization constants here for easy access ---
    # Based on your stats file: min=0.650, max=49.133
    gripper_min = 0.650288999080658
    gripper_max = 49.13294982910156
    # --- MODIFICATION END ---

    proprio = torch.zeros([1,14])

    control_step_counter = 0

    # 2frame buffer:with torch.inference_mode():
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        elif rdt:
            with torch.inference_mode():

                observation1 = robot.capture_observation()
                observation2 = robot.capture_observation()
                action = None
                img_arrs = [
                    observation1['observation.images.phone'],
                    observation1['observation.images.laptop'],
                    None,
                    observation2['observation.images.phone'],
                    observation2['observation.images.laptop'],
                    None
                ]
                images_pil = [PImage.fromarray(arr.cpu().numpy()) if arr is not None else None
                              for arr in img_arrs]

                # images_pil_train = load_training_images(episode_index=0, step_id=33, img_history_size=2)

                # train_current_state_radians = torch.tensor([[-0.1426,  3.4315,  3.1538,  0.952,  0.0092,  0.012]])
                # proprio[0, [7, 8, 9, 10, 11, 13]] = train_current_state_radians
                # curr_tatus = train_current_state_radians


                current_state_degrees = observation2['observation.state'].clone()  # (6,) tensor

                current_state_radians = current_state_degrees.clone()

                current_state_radians[:5] = torch.deg2rad(current_state_degrees[:5])

                current_state_radians[5] = (current_state_degrees[5] - gripper_min) / (gripper_max - gripper_min)

                proprio[0, [7, 8, 9, 10, 11, 13]] = current_state_radians

                curr_tatus = proprio[0, [7, 8, 9, 10, 11, 13]]
          #
          #       proprio_first = proprio
          #       # 直接创建一个 PyTorch 张量
          #       first_values = torch.tensor([[-0.1426,  3.4315,  3.1538,  0.952,  0.0092,  0.0059]])
          #       import numpy as np
          #       first_values[:, :5] = np.deg2rad(first_values[:, :5])
          #
          #       gripper_min = 0.650288999080658
          #       gripper_max = 49.13294982910156
          #       # The 6th column (index 5) is the gripper value
          #       first_values[:, 5] = (first_values[:, 5] - gripper_min) / (gripper_max - gripper_min)
          #
          #       proprio_first[0, [7, 8, 9, 10, 11, 13]] = first_values

                # 创建一个与真实lang_emd形状相同，但值全为0的张量
                # dummy_lang_emd = torch.zeros_like(lang_emd)

                actions = rdt_policy.step(
                    proprio=proprio,
                    images=images_pil,
                    text_embeds=lang_emd
                ).squeeze(0).cpu()

                # action_to_send_first34 = actions[:, [7, 8, 9, 10, 11, 13]]
                #
                # # 1. Convert predicted joint angles (first 5 values) from radians to degrees
                # action_to_send_first34[:, :5] = action_to_send_first34[:, :5] * 57.4
                #
                # # 2. De-normalize predicted gripper value (6th value) from [0, 1] to original range
                # action_to_send_first34[:, 5] = action_to_send_first34[:, 5] * (gripper_max - gripper_min) + gripper_min

                #import pdb; pdb.set_trace()

                # --- Postprocess output action ---
                # ... (your action postprocessing and robot command sending logic) ...

                # # ======================================================= #
                # # ==           ↓↓↓ 新增: WandB日志记录逻辑 ↓↓↓          == #
                # # ======================================================= #
                # if log_to_wandb and (control_step_counter % wandb_log_freq == 0):
                #     log_dict = {}
                #
                #     # 1. 记录输入图像 (t-1 和 t)
                #     # 我们只记录非空的图像
                #     if images_pil[0]: log_dict["inference/input_laptop_t-1"] = wandb.Image(images_pil[0])
                #     if images_pil[1]: log_dict["inference/input_phone_t-1"] = wandb.Image(images_pil[1])
                #     if images_pil[3]: log_dict["inference/input_laptop_t"] = wandb.Image(images_pil[3])
                #     if images_pil[4]: log_dict["inference/input_phone_t"] = wandb.Image(images_pil[4])
                #
                #     # 2. 记录输入的本体感受状态 (proprioception)
                #     # 我们记录的是预处理后、送入模型的最终状态
                #     proprio_to_log = proprio[0].cpu().numpy()
                #     proprio_table = wandb.Table(columns=[f"Joint_{i}" for i in range(5)] + ["Gripper"],
                #                                 data=[[proprio_to_log[i] for i in [7, 8, 9, 10, 11, 13]]])
                #     log_dict["inference/input_proprio"] = proprio_table
                #
                #     # 4. 记录模型输出的原始动作 (处理前)
                #     raw_action_np = actions.numpy()
                #     action_table = wandb.Table(columns=["Step"] + [f"Joint_{i}" for i in range(5)] + ["Gripper"],
                #                                data=[[i] + list(raw_action_np[i, [7, 8, 9, 10, 11, 13]]) for i in
                #                                      range(raw_action_np.shape[0])])
                #     log_dict["inference/output_raw_actions"] = action_table
                #
                #     # 5. 发送到 WandB，使用 control_step_counter 作为步数
                #     wandb.log(log_dict, step=control_step_counter)
                # # ======================================================= #

                control_step_counter += 1

                action_out = actions[:, [7, 8, 9, 10, 11, 13]]

                action_out = interp_action(curr_tatus, action_out)

                for mi in range(action_out.shape[0]):

                    action_to_send = action_out[mi, :].clone()  # (6,) tensor

                    action_to_send[:5] = action_to_send[:5] * 57.4

                    action_to_send[5] = action_to_send[5] * (gripper_max - gripper_min) + gripper_min

                    action = robot.send_action(action_to_send)

                    time.sleep(0.025)

                    action = {'action': action}

        else:
            observation = robot.capture_observation()
            action = None
            observation["task"] = [single_task]
            observation["robot_type"] = [policy.robot_type] if hasattr(policy, "robot_type") else [""]
            if policy is not None:
                pred_action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}

        if dataset is not None:
            observation = {k: v for k, v in observation.items() if k not in ["task", "robot_type"]}
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def load_training_images(episode_index=0, step_id=34, img_history_size=2):
    print(f"Loading training images for episode {episode_index} at step {step_id}...")

    # --- 1. 构建 episode 文件路径 ---
    HDF5_DIR = "data/datasets"
    DATASET_NAME = "overfit_dataset"
    file_name = f"episode_{episode_index:06d}.parquet"
    file_path = os.path.join(HDF5_DIR, DATASET_NAME, "data", "chunk-000", file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Episode file not found: {file_path}")

    def load_camera_frames(camera_name):
        video_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(file_path))),
            'videos',
            f"chunk-{str(episode_index // 1000).zfill(3)}",
            f"observation.images.{camera_name}"
        )
        video_path = os.path.join(video_dir, f"episode_{str(episode_index).zfill(6)}.mp4")

        if not os.path.exists(video_path):
            print(f"Warning: Video not found for camera {camera_name}, returning black frames.")
            black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            return [PImage.fromarray(black_frame)] * img_history_size

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = max(0, step_id - img_history_size + 1)
        end_frame = min(step_id + 1, total_frames)
        num_frames = end_frame - start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_np = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_np.append(frame)
        cap.release()

        if len(frames_np) < img_history_size:
            padding_count = img_history_size - len(frames_np)
            if frames_np:
                black_frame = np.zeros_like(frames_np[0])
            else:
                black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frames_np = [black_frame] * padding_count + frames_np
        # import pdb; pdb.set_trace()

        # 将 numpy 数组转换为 PIL Image 对象
        return [PImage.fromarray(f) for f in frames_np]

    top_view_frames = load_camera_frames("top_view")  # [top_t-1, top_t]
    hand_view_frames = load_camera_frames("hand_view")  # [hand_t-1, hand_t]

    # 检查 top_view_frames 和 hand_view_frames 是否包含两个元素
    if len(top_view_frames) != img_history_size or len(hand_view_frames) != img_history_size:
        raise ValueError(
            f"Expected {img_history_size} frames, but got {len(top_view_frames)} for top_view and {len(hand_view_frames)} for hand_view.")

    perfect_images_pil = [
        top_view_frames[0],   # ext_{t-1}
        hand_view_frames[0],  # right_wrist_{t-1}
        None,                 # left_wrist_{t-1} (固定为None)
        top_view_frames[1],   # ext_{t}
        hand_view_frames[1],  # right_wrist_{t}
        None                  # left_wrist_{t} (固定为None)
    ]

    print("Successfully loaded 'perfect' training images.")
    return perfect_images_pil

# In your rdt_inference_utils.py (or wherever control_loop is)
def control_loop_ground_truth(
        robot,
        gt_actions,  # 注意: 现在接收的是 (T, 6) 的 GT 动作序列
        control_time_s=None,
        events=None,
        fps: int | None = None,
):
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    timestamp = 0
    start_episode_t = time.perf_counter()

    # 确定我们要执行多少步
    num_gt_steps = gt_actions.shape[0]
    print(f"Starting ground truth execution for {num_gt_steps} steps.")

    # 循环遍历 GT 序列中的每一步
    for step_idx in range(num_gt_steps):
        start_loop_t = time.perf_counter()

        action_to_send = gt_actions[step_idx].clone()

        print(f"Step {step_idx}: Sending GT action (degrees): {action_to_send.numpy()}")

        robot.send_action(action_to_send)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break

        # 限制执行时间
        if timestamp >= control_time_s:
            break

    print("Ground truth execution finished.")

def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_data):
    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
