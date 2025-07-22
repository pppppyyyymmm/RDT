from lerobot.lerobot.common.robot_devices.robots.utils import make_robot_config, make_robot_from_config
from lerobot.lerobot.common.robot_devices.motors.feetech import TorqueMode

robot_new_config = make_robot_config('so101')
robot_new = make_robot_from_config(robot_new_config) # robot类

leader_arms = robot_new.leader_arms['main']
follower_arms = robot_new.follower_arms['main']

if not leader_arms.is_connected and not follower_arms.is_connected:
    print('CHECK the power, POWER OFF!')
