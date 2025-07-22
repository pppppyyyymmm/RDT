import sys
from pathlib import Path

import wandb
import time

# 获取当前脚本所在目录
current_dir = Path(__file__).resolve().parent

# 计算lerobot目录的路径（project/lerobot1/lerobot）
lerobot_dir = current_dir / "lerobot" 
print(lerobot_dir)
# 将lerobot目录添加到系统路径
sys.path.append(str(lerobot_dir))

from lerobot.lerobot.common.robot_devices.control_configs import (
    ControlConfig,
    ControlPipelineConfig,
    RDTControlConfig,
)
from lerobot.lerobot.common.robot_devices.control_utils import (
    control_loop, control_loop_ground_truth,
)
from lerobot.lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.lerobot.configs import parser
from data.hdf5_vla_dataset import HDF5VLADataset

import yaml
import numpy as np
import torch
from PIL import Image as PImage
import cv2

import os
import pandas as pd
import numpy as np
import torch

from scripts.agilex_model import create_model

# Names of cameras used for visual input
CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
observation_window = None
lang_embeddings = None

@safe_disconnect
def RDToperate(robot: Robot, cfg:RDTControlConfig, langembd, policy_in):
    # wandb.init(
    #     project="RDT_Inference_Debug", # 使用一个新项目名，或在老项目里用group区分
    #     name=f"real-robot-run-{int(time.time())}", # 给这次运行一个唯一的名字
    #     config={
    #         "robot_type": "so101",
    #         "policy_checkpoint": "checkpoints/rdt-finetune-170m-overfit-89/checkpoint-14000",
    #         "task": "pick and place"
    #     }
    # )
    action = control_loop(
            robot,
            control_time_s=cfg.reset_time_s,
            fps=cfg.fps,
            display_data=cfg.display_data,
            q_sim = None,
            rdt = True,
            rdt_policy = policy_in,
            lang_emd = langembd
        )

    # # 1. 加载一个 episode 的、连续的、6维的 GT 数据
    # gt_actions = load_ground_truth_episode(episode_index=0)
    #
    # # 2. 调用验证函数
    # control_loop_ground_truth(
    #     robot=robot,
    #     gt_actions=gt_actions, # 将 6 维的动作序列传入
    #     fps=10
    # )

    # wandb.finish()

@parser.wrap()
def control_real_robot(cfg: ControlPipelineConfig, argsin, config):

    # make a robot by your config, e.t so101_leader
    robot = make_robot_from_config(cfg.robot)

    # set a language task
    lang_ebd = get_lang(argsin)

    # import policy
    policy = make_policy(argsin)

    RDToperate(robot, cfg.control, lang_ebd, policy)

    if robot.is_connected:
        robot.disconnect()

def load_ground_truth_episode(episode_index=0):

    print(f"Directly loading and processing ground truth for episode {episode_index}...")

    # --- 1. 构建文件路径 ---
    HDF5_DIR = "data/datasets"
    DATASET_NAME = "overfit_dataset"
    file_name = f"episode_{episode_index:06d}.parquet"
    file_path = os.path.join(HDF5_DIR, DATASET_NAME, "data", "chunk-000", file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    # --- 2. 读取并处理 6 维数据 ---
    df = pd.read_parquet(file_path)
    actions = np.stack(df['action'].values)          # 形状: (T, 6)
    gt_actions = torch.from_numpy(actions).float()

    return gt_actions

def get_lang(argsin):
    lang_dict = torch.load(argsin['lang_embeddings_path']) # key-> embeddings  1 x 20 x 4096
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]
    return lang_embeddings

#Initialize the model
def make_policy(argsin):
    with open(argsin['config_path'], "r") as fp:
        config = yaml.safe_load(fp)
    argsin['config'] = config
    
    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=argsin['config'], 
        dtype=torch.bfloat16,
        pretrained=argsin['pretrained_model_name_or_path'],
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=argsin['ctrl_freq'],
    )

    return model

def get_config(args):
    config = {
        'episode_len': args['max_publish_step'],
        'state_dim': args['state_dim'],
        'chunk_size': args['chunk_size'],
        'camera_names': CAMERA_NAMES,
    }
    return config

# ctrl freq default 25
def arg_list():
    list_in = {'config_path' : "configs/base.yaml",\
    'max_publish_step' : 1000,\
    'state_dim' : 14,\
    'chunk_size' : 64,\
    'pretrained_model_name_or_path' : "./checkpoints/rdt-finetune-170m-new/checkpoint-34000",\
    'ctrl_freq' : 25, \
    'lang_embeddings_path' : 'outs/Pick_up_the_banana_and_place_it_on_the_plate.pt' }
    return list_in

def main():
    args = arg_list()
    config = get_config(args)    
    control_real_robot(args, config)
    # model_inference(args, config)

if __name__ == '__main__':
    main()
    # python run_simple.py --pretrained_model_name_or_path ckpt_1B/ --lang_embeddings_path outs/pick_banana.pt 

