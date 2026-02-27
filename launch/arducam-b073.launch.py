"""
dual_camera.launch.py

Launches the ArducamB0573Node.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def _make_node(context, *args, **kwargs):
    """
    OpaqueFunction callback — runs after all launch arguments are resolved.
    calib_url_left / calib_url_right are only added to the inline parameter dict
    when explicitly set by the user (non-empty).  When they are empty (default),
    the values from config/params.yaml are used unchanged.
    """
    pkg = FindPackageShare("isaac_ros_arducam_b0573").perform(context)
    params_yaml = os.path.join(pkg, "config", "params.yaml")

    node = Node(
        package="isaac_ros_arducam_b0573",
        executable="arducam_b0573_node",
        name="arducam_b0573_node",
        output="screen",
        parameters=[
            params_yaml
        ],
    )
    return [node]


def generate_launch_description():
    # ── Launch arguments ───────────────────────────────────────────────────
    

    return LaunchDescription([OpaqueFunction(function=_make_node)])
