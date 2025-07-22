from lerobot.lerobot.common.robot_devices.robots.utils import make_robot_config, make_robot_from_config
from lerobot.lerobot.common.robot_devices.motors.feetech import TorqueMode



robot_new_config = make_robot_config('so101')
robot_new = make_robot_from_config(robot_new_config) # robot类

leader_arms = robot_new.leader_arms['main']
follower_arms = robot_new.follower_arms['main']

if not leader_arms.is_connected:
    leader_arms.connect()
if not follower_arms.is_connected:
    follower_arms.connect() 


if (leader_arms.read('Torque_Enable') != TorqueMode.DISABLED.value).any():
    leader_arms.write("Torque_Enable", TorqueMode.DISABLED.value)

if (follower_arms.read('Torque_Enable') != TorqueMode.DISABLED.value).any():
    follower_arms.write("Torque_Enable", TorqueMode.DISABLED.value)

