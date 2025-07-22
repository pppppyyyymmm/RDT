import logging
import os
import time
from dataclasses import asdict
from pprint import pformat

import rerun as rr


# import sys
# from pathlib import Path

# # 获取当前脚本所在目录
# current_dir = Path(__file__).resolve().parent


# # 计算lerobot目录的路径（project/lerobot1/lerobot）
# lerobot_dir = current_dir / "lerobot" / "lerobot"

# # 将lerobot目录添加到系统路径
# sys.path.append(str(lerobot_dir))



# from safetensors.torch import load_file, save_file
from lerobot.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lerobot.common.policies.factory import make_policy
from lerobot.lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
    SimoperateControlConfig,
)
from lerobot.lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    is_headless,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

import queue
import threading
import time

@safe_disconnect
def simoperate(robot: Robot, cfg:SimoperateControlConfig, queue_in):
    action = control_loop(
            robot,
            control_time_s=cfg.teleop_time_s,
            fps=cfg.fps,
            # teleoperate=True,
            display_data=cfg.display_data,
            simoperate=True,
            q_sim = queue_in
        )
    queue_in.put(None)

@parser.wrap()
def control_sim_robot(cfg: ControlPipelineConfig, queue_leader):

    # make a robot by your config, e.t so101_leader
    robot = make_robot_from_config(cfg.robot)

    if isinstance(cfg.control, SimoperateControlConfig):
        simoperate(robot, cfg.control, queue_leader)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()

def issac_load(queue):
    while True:
        action_sim = queue.get()
        if action_sim is None:
            break
        print(f'read action {action_sim}')
        queue.task_done()


if __name__ == "__main__":
    q = queue.Queue()

    # one thread for lerobot write motor action
    producer_thread = threading.Thread(target=control_sim_robot,args=(q,))
    producer_thread.start()

    # one thread for isaacsim read action data
    isaac_load_thread = threading.Thread(target=issac_load, args=(q,))
    isaac_load_thread.start()

    producer_thread.join()
    isaac_load_thread.join()
