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

  * @file nitron_image_bridge.launch.py
  * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
  * @date: 27.02.2026
  * @brief: Converts NITROS images to RGB8 using isaac_ros_image_proc's ImageFormatConverterNode.
            This bridge is required because the ArducamB0573Node publishes in NV12 format.
            The resulting RGB8 images are published on topics:
            - /arducam/left/image_rgb
            - /arducam/right/image_rgb
----------------------------------------------------------------------------------------------
"""


# Convert arducam NITROS images → RGB8 using isaac_ros_image_proc's
# ImageFormatConverterNode (GPU-accelerated, zero-copy NITROS pipeline).
#
# The source format is whatever the camera node publishes, controlled by:
#   left_camera.topics.nitros_image.format  (params.yaml)
#
# Recommended: set format = "rgb8" in params.yaml.
#   - TypeAdapter correctly copies all pixel data (step = width*3).
#   - ImageFormatConverterNode acts as a passthrough (rgb8 → rgb8).
#   - The bridge topics /arducam/{left,right}/image_rgb are still published.
#
# Avoid format = "nv12" with this bridge: the TypeAdapter's convert_to_custom
# uses a single cudaMemcpy2D that copies only the Y plane into the GXF
# VideoBuffer; UV remains uninitialized → VPI NV12→RGB sees random chroma
# and may crash or produce garbage output.
#
# Topics produced:
#   /arducam/left/image_rgb   (sensor_msgs/Image  rgb8)
#   /arducam/right/image_rgb  (sensor_msgs/Image  rgb8)

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

# Must match left/right_camera.topics.nitros_image.resolution.{width,height} in params.yaml.
# Defaults to eye_width × combined_height when -1 is set in params (e.g. 1280 × 720).
IMAGE_WIDTH  = 640
IMAGE_HEIGHT = 480


def generate_launch_description():

    left_converter = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_converter_left',
        parameters=[{
            'encoding_desired': 'rgb8',
            'image_width':  IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
        }],
        remappings=[
            ('image_raw', '/arducam/left/nitros_image_nv12'),
            ('image',     '/arducam/left/image_rgb'),
        ],
    )

    right_converter = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_converter_right',
        parameters=[{
            'encoding_desired': 'rgb8',
            'image_width':  IMAGE_WIDTH,
            'image_height': IMAGE_HEIGHT,
        }],
        remappings=[
            ('image_raw', '/arducam/right/nitros_image_nv12'),
            ('image',     '/arducam/right/image_rgb'),
        ],
    )

    container = ComposableNodeContainer(
        name='nitros_image_bridge_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            left_converter,
            right_converter,
        ],
        output='screen',
    )

    return LaunchDescription([container])
