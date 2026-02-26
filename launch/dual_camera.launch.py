"""
dual_camera.launch.py

Launches the ArducamDualCamNode.

Usage examples:
  # Default  (2560×720 @ 30 fps):
  ros2 launch arducam_dual_camera dual_camera.launch.py

  # High-res:
  ros2 launch arducam_dual_camera dual_camera.launch.py width:=3840 height:=1200

  # Low-res, higher FPS:
  ros2 launch arducam_dual_camera dual_camera.launch.py width:=1280 height:=480 fps:=60

  # Override calibration URLs at launch time:
  ros2 launch arducam_dual_camera dual_camera.launch.py \\
    calib_url_left:=file:///path/to/left.yaml \\
    calib_url_right:=file:///path/to/right.yaml

NOTE on calibration URL precedence:
  calib_url_left / calib_url_right are read from config/params.yaml by default.
  Passing them as launch arguments here overrides that value.
  Leaving the launch argument at its default ("keep_from_yaml") means params.yaml wins.
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
    pkg = FindPackageShare("arducam_dual_camera").perform(context)
    params_yaml = os.path.join(pkg, "config", "params.yaml")

    # Parameters that always come from launch args (override yaml)
    inline_params = {
        "device":            LaunchConfiguration("device").perform(context),
        "width":             int(LaunchConfiguration("width").perform(context)),
        "height":            int(LaunchConfiguration("height").perform(context)),
        "fps":               int(LaunchConfiguration("fps").perform(context)),
        "pixel_format":      LaunchConfiguration("pixel_format").perform(context),
        "frame_id_left":     LaunchConfiguration("frame_id_left").perform(context),
        "frame_id_right":    LaunchConfiguration("frame_id_right").perform(context),
        "camera_namespace":  LaunchConfiguration("camera_namespace").perform(context),
        "camera_name_left":  LaunchConfiguration("camera_name_left").perform(context),
        "camera_name_right": LaunchConfiguration("camera_name_right").perform(context),
        "qos.reliability":   LaunchConfiguration("qos_reliability").perform(context),
        "qos.durability":    LaunchConfiguration("qos_durability").perform(context),
    }

    # Only override calib URLs when the user explicitly passes them
    calib_left  = LaunchConfiguration("calib_url_left").perform(context)
    calib_right = LaunchConfiguration("calib_url_right").perform(context)
    if calib_left:
        inline_params["calib_url_left"]  = calib_left
    if calib_right:
        inline_params["calib_url_right"] = calib_right

    node = Node(
        package="arducam_dual_camera",
        executable="arducam_dual_cam_node",
        name="arducam_dual_cam_node",
        output="screen",
        parameters=[
            params_yaml,
            inline_params,
        ],
    )
    return [node]


def generate_launch_description():
    # ── Launch arguments ───────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("device",         default_value="/dev/video0",
                              description="V4L2 device node"),
        DeclareLaunchArgument("width",          default_value="2560",
                              description="Combined (side-by-side) frame width. "
                                          "Available: 3840, 2560, 1280"),
        DeclareLaunchArgument("height",         default_value="720",
                              description="Frame height. Available: 1200, 720, 480"),
        DeclareLaunchArgument("fps",            default_value="30",
                              description="Target frame rate (0 = driver default)"),
        DeclareLaunchArgument("pixel_format",   default_value="UYVY",
                              description="Input pixel format: UYVY or NV16"),
        DeclareLaunchArgument("frame_id_left",  default_value="left_camera",
                              description="TF frame ID for the left camera"),
        DeclareLaunchArgument("frame_id_right", default_value="right_camera",
                              description="TF frame ID for the right camera"),
        DeclareLaunchArgument("camera_name_left",  default_value="left",
                              description="Camera name for camera_info_manager (left)"),
        DeclareLaunchArgument("camera_name_right", default_value="right",
                              description="Camera name for camera_info_manager (right)"),
        DeclareLaunchArgument("camera_namespace",  default_value="arducam",
                              description="Topic namespace prefix: /<namespace>/left/image_raw etc."),
        DeclareLaunchArgument("qos_reliability",   default_value="best_effort",
                              description="Publisher QoS reliability: best_effort | reliable"),
        DeclareLaunchArgument("qos_durability",    default_value="volatile",
                              description="Publisher QoS durability: volatile | transient_local"),
        DeclareLaunchArgument(
            "calib_url_left",  default_value="",
            description="Calibration URL for left camera "
                        "(e.g. file:///path/to/left.yaml). "
                        "Empty = use value from params.yaml."),
        DeclareLaunchArgument(
            "calib_url_right", default_value="",
            description="Calibration URL for right camera. "
                        "Empty = use value from params.yaml."),
    ]

    return LaunchDescription(args + [OpaqueFunction(function=_make_node)])
