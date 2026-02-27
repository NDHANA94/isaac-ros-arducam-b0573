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

  * @file list_modes.launch.py
  * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
  * @date: 27.02.2026
  * @brief: Runs the list_camera_modes.py utility script to enumerate all
            V4L2 formats+resolutions available on the Arducam device.
----------------------------------------------------------------------------------------------
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
