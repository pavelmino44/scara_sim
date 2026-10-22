"""
Запуск управляемого движения (ПИД-регулятор).

    ros2 launch scara_sim pid_controller.launch.py                 # ДПТ, ff+pid
    ros2 launch scara_sim pid_controller.launch.py mode:=pid
    ros2 launch scara_sim pid_controller.launch.py mode:=ff
    ros2 launch scara_sim pid_controller.launch.py actuator:=torque  # идеальный привод
    ros2 launch scara_sim pid_controller.launch.py gains:=/path/my_gains.yaml motors:=/path/my_motors.yaml

actuator:=motor  -> pid_controller (напряжение) + dc_motor_sim (ДПТ -> момент)
actuator:=torque -> pid_controller (момент сразу в Gazebo)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):

    pkg = get_package_share_directory('scara_sim')

    actuator = context.launch_configurations['actuator']
    mode = context.launch_configurations['mode']
    motors = context.launch_configurations['motors']
    gains = context.launch_configurations['gains']

    if not gains:
        gains = os.path.join(
            pkg, 'config',
            'pid_gains.yaml' if actuator == 'motor' else 'pid_gains_torque.yaml')

    common = {'use_sim_time': True}

    nodes = []

    if actuator == 'motor':
        nodes.append(Node(
            package='scara_sim',
            executable='dc_motor_sim',
            name='dc_motor_sim',
            output='screen',
            parameters=[motors, gains, common],
        ))
        controller_params = [motors, gains, common, {'mode': mode}]
    else:
        controller_params = [gains, common, {'mode': mode, 'actuator': 'torque'}]

    nodes.append(Node(
        package='scara_sim',
        executable='pid_controller',
        name='pid_controller',
        output='screen',
        parameters=controller_params,
    ))

    return nodes


def generate_launch_description():

    pkg = get_package_share_directory('scara_sim')

    return LaunchDescription([
        DeclareLaunchArgument(
            'actuator', default_value='motor', choices=['motor', 'torque'],
            description='motor - ПИД управляет ДПТ (напряжение), torque - момент напрямую'),
        DeclareLaunchArgument(
            'mode', default_value='ff+pid', choices=['ff', 'pid', 'ff+pid'],
            description='ff - только прямая связь, pid - только ПИД, ff+pid - оба'),
        DeclareLaunchArgument(
            'gains', default_value='',
            description='YAML с коэффициентами (пусто -> по actuator)'),
        DeclareLaunchArgument(
            'motors', default_value=os.path.join(pkg, 'config', 'dc_motors.yaml'),
            description='YAML с параметрами ДПТ'),
        OpaqueFunction(function=launch_setup),
    ])
