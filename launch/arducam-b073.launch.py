"""
  MIT License -------------------------------------------------------------------------------
  Copyright (c) 2026 W.M. Nipun Dhananjaya

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

  --------------------------------------------------------------------------------------------

  * @file dual_camera.cpp
  * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
  * @date: 27.02.2026
  * @brief Launches the ArducamB0573Node.
----------------------------------------------------------------------------------------------
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
