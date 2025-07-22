import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np
import pandas as pd

from configs.state_vec import STATE_VEC_IDX_MAPPING
Episode_Num = 80

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        # 修改为你的数据集路径
        self.HDF5_DIR = "data/datasets"
        self.DATASET_NAME = "so101_2cam_banana_240x320_opencv"

        # 初始化文件路径列表
        self.file_paths = []

        # 生成并存储所有文件路径
        self.generate_file_paths()

    def generate_file_paths(self):
        base_path = os.path.join(self.HDF5_DIR, self.DATASET_NAME, "data", "chunk-000")

        for i in range(Episode_Num):
            file_name = f"episode_{i:06d}.parquet"
            file_path = os.path.join(base_path, file_name)

            rel_path = os.path.join(self.HDF5_DIR, self.DATASET_NAME, "data", "chunk-000", file_name)
            self.file_paths.append(rel_path)

        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def parse_hdf5_file(self, file_path):
        # 读取 parquet 文件`
        df = pd.read_parquet(file_path)
        qpos = np.stack(df['observation.state'].values)  # 形状: (T, 6)
        actions = np.stack(df['action'].values)  # 形状: (T, 6)
        #import pdb; pdb.set_trace()

        # --- Convert degrees to radians ---
        # The first 5 columns are joint angles, the 6th is the gripper.
        # Only convert the joint angles.
        qpos[:, :5] = np.deg2rad(qpos[:, :5])
        actions[:, :5] = np.deg2rad(actions[:, :5])
        # --- MODIFICATION END ---

        # --- Normalize gripper to [0, 1] ---
        # Based on your stats file: min=0.650, max=49.133 for state
        # We use the same stats for actions for consistency
        gripper_min = 0.650288999080658
        gripper_max = 49.13294982910156
        # The 6th column (index 5) is the gripper value
        qpos[:, 5] = (qpos[:, 5] - gripper_min) / (gripper_max - gripper_min)
        actions[:, 5] = (actions[:, 5] - gripper_min) / (gripper_max - gripper_min)
        # --- NEW MODIFICATION END ---

        num_steps = qpos.shape[0]
        # [Optional] We drop too-short episode
        if num_steps < 128:
            return False, None

        # 跳过静止帧（与原始逻辑一致）
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")

        # 随机采样时间步
        start_idx = max(0, first_idx - 1)
                # --- MODIFICATION END ---
        step_id = np.random.randint(start_idx, num_steps)

        # 加载语言指令（从 meta/ 目录，这里先直接赋值
        episode_idx = int(os.path.basename(file_path).split('_')[1].split('.')[0])
        task_desc = "Pick up the banana and place it on the plate."
        instruction = 'data/datasets/so101_2cam_banana_240x320_opencv/meta/lang_embed_0.pt'
        # instruction = task_desc

        # import pdb; pdb.set_trace()
        # Assemble the meta
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction
        }

        # Parse the state and action
        state = qpos[step_id:step_id + 1]
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
        actions = actions[step_id:step_id+self.CHUNK_SIZE]
        if actions.shape[0] < self.CHUNK_SIZE:
            # Pad the actions using the last action
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))
            ], axis=0)

        # 填充到统一状态向量
        def fill_in_state(values):
            UNI_STATE_INDICES = [
                                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(5)
                                ] + [
                                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                                ]

            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        state = fill_in_state(state)
        state_indicator = fill_in_state(np.ones_like(state_std))
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm = fill_in_state(state_norm)
        # If action's format is different from state's,
        # you may implement fill_in_action()
        actions = fill_in_state(actions)

        # 替换 parse_img 函数
        def load_camera_frames(camera_name, step_id, first_idx):
            """加载指定摄像头的图像历史帧"""
            # 1. 构建视频文件路径
            video_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(file_path))),  # 上两级目录
                'videos',
                f"chunk-{str(episode_idx // 1000).zfill(3)}",  # chunk 目录
                f"observation.images.{camera_name}"  # 摄像头类型
            )
            video_path = os.path.join(video_dir, f"episode_{str(episode_idx).zfill(6)}.mp4")

            # 2. 如果视频文件不存在，返回空数组
            if not os.path.exists(video_path):
                return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)), np.zeros(self.IMG_HISORY_SIZE, dtype=bool)

            # 3. 使用 OpenCV 打开视频
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 4. 计算需要加载的帧范围
            start_frame = max(0, step_id - self.IMG_HISORY_SIZE + 1)
            end_frame = min(step_id + 1, total_frames)  # +1 因为 range 不包含结束值
            num_frames = end_frame - start_frame

            # 5. 跳转到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # 6. 读取帧并转换为RGB
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            # 7. 如果帧数不足，在前面填充黑色帧
            if len(frames) < self.IMG_HISORY_SIZE:
                padding_frames = self.IMG_HISORY_SIZE - len(frames)
                if frames:
                    black_frame = np.zeros_like(frames[0])
                else:
                    # 如果没有帧，创建默认尺寸 (240x320x3)
                    black_frame = np.zeros((240, 320, 3), dtype=np.uint8)

                frames = [black_frame] * padding_frames + frames

            # 8. 转换为 numpy 数组
            frames_array = np.array(frames)

            # 9. 创建有效帧掩码
            # 有效帧：从第一个有效帧(first_idx-1)到当前帧
            valid_start = max(start_frame, first_idx - 1)
            valid_in_sequence = max(0, valid_start - start_frame)
            valid_count = min(num_frames, step_id - valid_start + 1)

            # 10. 创建掩码数组
            mask = np.zeros(self.IMG_HISORY_SIZE, dtype=bool)
            mask[valid_in_sequence:valid_in_sequence + valid_count] = True

            return frames_array, mask

        # 加载顶部摄像头（外部视角）
        cam_high, cam_high_mask = load_camera_frames("top_view", step_id, first_idx)

        # 加载手腕摄像头（SO101只有一个手腕摄像头，视为右手腕）
        cam_right_wrist, cam_right_wrist_mask = load_camera_frames("hand_view", step_id, first_idx)

        # 左手腕摄像头不可用
        cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
        cam_left_wrist_mask = np.zeros(self.IMG_HISORY_SIZE, dtype=bool)

        # Return the resulting sample
        # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
        # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
        # if the left-wrist camera is unavailable on your robot
        return True, {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_high_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_left_wrist_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask
        }

    def parse_hdf5_file_state_only(self, file_path):
        # 使用 pandas 读取 parquet 文件
        df = pd.read_parquet(file_path)
        qpos = np.stack(df['observation.state'].values)  # 形状: (T, 6)
        actions = np.stack(df['action'].values)  # 形状: (T, 6)

        # --- Convert degrees to radians ---
        # The first 5 columns are joint angles, the 6th is the gripper.
        # Only convert the joint angles.
        qpos[:, :5] = np.deg2rad(qpos[:, :5])
        actions[:, :5] = np.deg2rad(actions[:, :5])
        # --- MODIFICATION END ---

        # --- Normalize gripper to [0, 1] ---
        gripper_min = 0.650288999080658
        gripper_max = 49.13294982910156
        qpos[:, 5] = (qpos[:, 5] - gripper_min) / (gripper_max - gripper_min)
        actions[:, 5] = (actions[:, 5] - gripper_min) / (gripper_max - gripper_min)
        # --- NEW MODIFICATION END ---

        # 跳过初始静止帧
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0])
        moving_indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]

        if moving_indices.size > 0:
            first_idx = moving_indices[0]
        else:
            # 如果没有移动帧，使用整个轨迹
            first_idx = 0

        # 提取有效部分（跳过静止帧）
        state = qpos[first_idx:]
        action = actions[first_idx:]

        # 填充到统一状态向量
        def fill_in_state(values):
            UNI_STATE_INDICES = [
                                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(5)
                                ] + [
                                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                                ]

            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec

        return True, {
            "state": fill_in_state(state),
            "action": fill_in_state(action)
        }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
