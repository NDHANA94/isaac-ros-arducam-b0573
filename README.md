# arducam_dual_camera

ROS 2 Humble package for the **Arducam B0573** — a 2.3 MP Global Shutter Dual-Camera kit (GMSL2-to-CSI2) on NVIDIA Jetson Orin Nano.

Captures a side-by-side stereo frame from the CSI-2 port, splits it into independent left and right images using hardware-accelerated format conversion, and publishes both as standard `sensor_msgs/Image` topics at configurable resolution and frame rate.

---

## Hardware Setup

```
[ Left Camera ]──┐
                  ├── GMSL2 link ──► Arducam B0573 GMSL2-to-CSI2 Serializer/Deserializer ──► CSI-2 Port 2 (Jetson Orin Nano)
[ Right Camera ]─┘
```

- **Camera**: Arducam B0573 (OV2311 sensor, global shutter, 1600×1200 native per eye)
- **Interface**: GMSL2 → CSI-2, enumerated by the Jetson tegra-video V4L2 driver
- **Device node**: `/dev/video0`  (`tegra-capture-vi:2`, arducam-csi2 9-000c)
- **Driver**: `tegra-video` kernel driver (JetPack 6.2.2)

---

## Architecture & Pipeline

### Full data-flow diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Jetson Orin Nano                                                           │
│                                                                             │
│  CSI-2 Port 2                                                               │
│      │                                                                      │
│      ▼                                                                      │
│  ┌──────────┐   UYVY/NV16      ┌────────────┐   BGRx (4 B/px)             │
│  │ v4l2src  │ ───────────────► │ nvvidconv  │ ──────────────────────────┐  │
│  │ (kernel  │   DMA buffer     │ (Jetson    │                           │  │
│  │  driver) │                  │  VIC HW)   │                           │  │
│  └──────────┘                  └────────────┘                           │  │
│                                                                          │  │
│                                               ┌──────────────────────── ▼ ─┤
│                                               │        appsink           │  │
│                                               │  (GStreamer pull model)  │  │
│                                               └────────────┬─────────────┘  │
│                                                            │                │
│                                    gst_buffer_map (READ)  │  map.data ptr  │
│                                                            ▼                │
│                                          ┌──────────────────────────────┐  │
│                                          │  cv::Mat combined            │  │
│                                          │  (3840×1200 / 2560×720 /     │  │
│                                          │   1280×480, CV_8UC4, BGRx)   │  │
│                                          │  ── NO pixel copy ──         │  │
│                                          └──────────┬───────────────────┘  │
│                                                     │  ROI split (x=0..W/2, x=W/2..W)
│                                          ┌──────────┴────────────┐         │
│                                          │                        │         │
│                                   cv::Mat left              cv::Mat right   │
│                                   (ROI, no copy)            (ROI, no copy)  │
│                                          │                        │         │
│                              cvtColor BGRA→BGR          cvtColor BGRA→BGR  │
│                                          │                        │         │
│                          ┌───────────────┘        ┌───────────────┘         │
│                          ▼                        ▼                         │
│              ┌───────────────────┐    ┌───────────────────┐                │
│              │ image_transport   │    │ image_transport   │                │
│              │ publisher (left)  │    │ publisher (right) │                │
│              └────────┬──────────┘    └────────┬──────────┘                │
│                       │                        │                           │
└───────────────────────┼────────────────────────┼───────────────────────────┘
                        │                        │
          ┌─────────────▼──────┐    ┌────────────▼─────────┐
          │ /arducam/left/     │    │ /arducam/right/      │
          │   image_raw        │    │   image_raw          │
          │   camera_info      │    │   camera_info        │
          └────────────────────┘    └──────────────────────┘
```

### Hardware acceleration stages

| Stage | Hardware unit | What happens |
|---|---|---|
| CSI-2 capture | NVCSI + VI (Video Input) | Raw Bayer/YUV pixels DMA'd directly into a buffer — no CPU involvement |
| Format conversion | **VIC (Video Image Compositor)** via `nvvidconv` | UYVY → BGRx in hardware; same unit used by `isaac_ros_argus_camera` |
| Frame split | CPU (cv::Mat ROI) | Only a pointer offset — **zero pixel copy** for the split itself |
| BGRA→BGR channel drop | CPU (OpenCV `cvtColor`) | Single-pass, removes the unused X channel |
| Publish | CPU | `cv_bridge` wraps the Mat into a ROS message header |

> **Why not Argus?**  
> `nvarguscamerasrc` requires an ISP-compatible sensor registered in the Argus sensor database. The Arducam B0573 exposes itself as a generic V4L2 device (`tegra-video` driver), bypassing the Argus ISP stack. The `v4l2src + nvvidconv` path still uses the same VIC hardware for format conversion, giving equivalent zero-CPU-decode performance.

---

## Topics Published

All topics use `sensor_msgs/Image` (encoding: `bgr8`) and `sensor_msgs/CameraInfo`.

| Topic | Type | Description |
|---|---|---|
| `/arducam/left/image_raw` | `sensor_msgs/Image` | Left camera, BGR8 |
| `/arducam/left/camera_info` | `sensor_msgs/CameraInfo` | Left intrinsics |
| `/arducam/right/image_raw` | `sensor_msgs/Image` | Right camera, BGR8 |
| `/arducam/right/camera_info` | `sensor_msgs/CameraInfo` | Right intrinsics |

> `camera_info` ships with placeholder identity intrinsics. Replace with values from a proper stereo calibration (`camera_calibration` ROS package) for metric 3-D use.

---

## Available Resolutions & Frame Rates

The B0573 reports three discrete modes via V4L2 (`/dev/video0`):

| Combined frame | Per-eye resolution | Pixel formats |
|---|---|---|
| 3840 × 1200 | **1920 × 1200** | UYVY, NV16 |
| 2560 × 720  | **1280 × 720** ← default | UYVY, NV16 |
| 1280 × 480  | **640 × 480** | UYVY, NV16 |

> Frame-rate intervals are negotiated by the tegra-video driver and not enumerated via `VIDIOC_ENUM_FRAMEINTERVALS`. The driver honours the `framerate=N/1` GStreamer caps if the sensor supports it. Default is 30 fps; try 60 fps at 1280×480.

To list all V4L2 modes at runtime:

```bash
ros2 run arducam_dual_camera list_camera_modes.py
# or directly:
python3 src/arducam_dual_camera/scripts/list_camera_modes.py /dev/video0
```

---

## Parameters

All parameters are settable via `params.yaml` or as launch arguments.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `device` | string | `/dev/video0` | V4L2 device node |
| `width` | int | `2560` | **Combined** frame width (left+right) |
| `height` | int | `720` | Frame height |
| `fps` | int | `30` | Target frame rate (0 = driver default) |
| `pixel_format` | string | `UYVY` | V4L2 input format: `UYVY` or `NV16` |
| `frame_id_left` | string | `left_camera` | TF frame ID for left camera |
| `frame_id_right` | string | `right_camera` | TF frame ID for right camera |

---

## Build

```bash
cd /home/orin/workspace/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select arducam_dual_camera
source install/setup.bash
```

> The OpenCV linker warnings (`libopencv_imgproc.so.4.5d` vs `.408`) are benign — a known JetPack 6 / ROS Humble version mismatch. The node runs correctly.

---

## Run

```bash
source install/setup.bash

# Default: 2560×720 @ 30 fps  → 1280×720 per eye
ros2 launch arducam_dual_camera dual_camera.launch.py

# High resolution: 3840×1200 → 1920×1200 per eye
ros2 launch arducam_dual_camera dual_camera.launch.py \
  width:=3840 height:=1200

# Low resolution: 1280×480 → 640×480 per eye
ros2 launch arducam_dual_camera dual_camera.launch.py \
  width:=1280 height:=480 fps:=60

# Custom device
ros2 launch arducam_dual_camera dual_camera.launch.py device:=/dev/video2
```

### Verify output topics

```bash
# List active topics
ros2 topic list | grep arducam

# Check frame rate
ros2 topic hz /arducam/left/image_raw

# View in rqt
ros2 run rqt_image_view rqt_image_view /arducam/left/image_raw &
ros2 run rqt_image_view rqt_image_view /arducam/right/image_raw
```

---

## Package Structure

```
src/arducam_dual_camera/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── params.yaml                     # Default node parameters
├── include/arducam_dual_camera/
│   └── arducam_dual_cam_node.hpp       # Node class declaration
├── launch/
│   ├── dual_camera.launch.py           # Main launch (all params exposed)
│   └── list_modes.launch.py            # Utility: print V4L2 modes
├── scripts/
│   └── list_camera_modes.py            # V4L2 mode/FPS enumerator
└── src/
    └── arducam_dual_cam_node.cpp       # Node implementation
```

---

## Integration with Isaac ROS

The published `sensor_msgs/Image` topics can be fed directly into Isaac ROS nodes:

```python
# Example: wire into isaac_ros_yolov8
# Remap /arducam/left/image_raw → /image (or wherever the detector expects input)
ros2 launch isaac_ros_yolov8 yolov8_core.launch.py \
  input_image_width:=1280 input_image_height:=720 \
  ...
```

For zero-copy NITROS handoff, an `isaac_ros_nitros_bridge` can be inserted between this node's output and the Isaac inference pipeline.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Failed to build GStreamer pipeline` | Wrong device path or camera not powered | Check `v4l2-ctl --list-devices`; ensure GMSL2 cable is seated |
| `Buffer size mismatch` warning | `nvvidconv` outputting unexpected format | Try `pixel_format:=NV16` or verify caps with `gst-launch-1.0 v4l2src device=/dev/video0 ! nvvidconv ! fakesink -v` |
| Image is fully black / one side is garbled | Combined width set incorrectly | Confirm via `v4l2-ctl -d /dev/video0 --all`; `width` param must match `Width/Height` reported |
| Low FPS at high resolution | CPU `cvtColor` bottleneck | Consider `NV16 → NV12` path and use `cv::cuda::cvtColor` for the channel conversion |
| `nvvidconv` not found | GStreamer Jetson plugins not installed | `sudo apt install gstreamer1.0-plugins-bad` and ensure `libgstreamer-plugins-bad1.0-dev` |
