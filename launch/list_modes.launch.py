"""
list_modes.launch.py
Runs the list_camera_modes.py utility script to enumerate all
V4L2 formats+resolutions available on the Arducam device.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("device", default_value="/dev/video0"),
        ExecuteProcess(
            cmd=["python3", "-c",
                 "import subprocess, sys; "
                 "subprocess.run(['v4l2-ctl', '-d', sys.argv[1], "
                 "'--list-formats-ext'], check=True)",
                 LaunchConfiguration("device")],
            output="screen",
        ),
    ])
