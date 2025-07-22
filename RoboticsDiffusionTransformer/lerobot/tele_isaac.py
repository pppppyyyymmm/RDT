# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import sys
import time
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# preparing the scene
assets_root_path = get_assets_root_path()
# assets_root_path = '~/isaacsim/lerobot/robotbody'

print(assets_root_path)
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  # add ground plane
set_camera_view(
    eye=[2.0, -0.5, 1.0], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp" # 5.0 0.0 1.5
)  # set camera view (eye: [far,near],[horizental rotate -left,+right], [vertically rotate +up -down])

# # Add Franka
# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
# add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")  # add robot to stage
# arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")  # create an articulation object



# # Add so101
# asset_path = assets_root_path + "/so101_new_calib/so101_new_calib.usd"
# add_reference_to_stage(usd_path=asset_path, prim_path="/World")  # add robot to stage
# arm = Articulation(prim_paths_expr="/World", name="my_101")  # create an articulation object


# set the initial poses of the arm and the car so they don't collide BEFORE the simulation starts
arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())

# initialize the world
my_world.reset()

# for i in range(4):
#     print("running cycle: ", i)
#     if i == 1 or i == 3:
#         print("moving")
#         # move the arm
#         arm.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])
#     if i == 2:
#         print("stopping")
#         # reset the arm
#         arm.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

#     for j in range(10):
#         # step the simulation, both rendering and physics
#         my_world.step(render=True)
#         # print the joint positions of the car at every physics step
#         # if i == 3:
#         #     car_joint_positions = car.get_joint_positions()
#         #     print("car joint positions:", car_joint_positions)

# time.sleep(5)

simulation_app.close()
