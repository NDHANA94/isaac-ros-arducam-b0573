# arducam_dual_camera

A high-performance ROS 2 Humble composable node for the **Arducam B0573** — a 2.3 MP Global Shutter Dual-Camera kit (GMSL2-to-CSI2) — running on NVIDIA Jetson Orin Nano.

The node captures a hardware-synchronized side-by-side (SBS) stereo frame from the CSI-2 port, splits it into independent left and right streams, and publishes them over three configurable topic channels per camera:

| Channel | Topic (default) | Message type | Notes |
|---|---|---|---|
| `visual_stream` | `…/image/compressed` | `sensor_msgs/CompressedImage` | JPEG, network-friendly (default) |
| `visual_stream` | `…/image_raw` | `sensor_msgs/Image` | raw BGR/RGB (set `transport: raw`) |
| `nitros_image` | `…/nitros_image_nv12` | `NitrosImage` (NV12) | GPU-resident zero-copy, HAVE_NVBUF only |
| `camera_info` | `…/camera_info` | `sensor_msgs/CameraInfo` | calibration, always published |

---

## Hardware

```
[ OV2311 Left  ]──┐                        ┌─ /arducam/left/*
                   ├─ GMSL2 ─► B0573 ─► CSI-2 (Jetson)
[ OV2311 Right ]──┘            (SBS frame) └─ /arducam/right/*
```

| Property | Value |
|---|---|
| Sensor | OV2311 Global Shutter (1600×1200 native per eye) |
| Interface | GMSL2 → CSI-2 single cable, `/dev/video0` |
| Platform | NVIDIA Jetson Orin Nano (JetPack 6.2+, DeepStream 7.1+) |
| Synchronization | Hardware-locked — both eyes share the same V4L2 buffer PTS |

---

## Pipeline & Data Flow

### Capture pipeline (GStreamer)

Three pipeline variants are tried on startup in order of preference:

```
1. v4l2src(UYVY) → nvvidconv/VIC → NV12(NVMM)  → appsink  [preferred — 1.5 B/px]
2. v4l2src(UYVY) → nvvidconv/VIC → BGRx(NVMM)  → appsink  [NVMM fallback — 4 B/px]
3. v4l2src(UYVY) → nvvidconv/VIC → BGRx         → appsink  [system-memory CPU path]
```

`nvvidconv` always runs on the **VIC** (Video Image Compositor) hardware engine, so the UYVY→NV12 colour-space conversion is zero-CPU regardless of which variant is active.

### Per-frame processing (HAVE_NVBUF path — preferred)

```
gst_buffer_map(NVMM)                  ← O(1), returns NvBufSurface* pointer only
       │
       ├── NvBufSurfTransformAsync ×2  ← VIC: NV12 crop → nvbuf_left_  / nvbuf_right_
       │                                  (SURFACE_ARRAY dst, required by VIC on Orin)
       ├── NvBufSurfTransformAsync ×2  ← VIC: NV12→BGRA crop → nvbuf_raw_left_ / _right_
       │                                  (all 4 VIC jobs dispatched concurrently)
       │
       ├── SyncObjWait ×4             ← wait for all VIC jobs; gst_buffer_unmap
       │
       ├── NvBufSurfaceSyncForCpu ×2  ← make CPU view coherent (NV12 SURFACE_ARRAY)
       │
       ├── cudaMemcpy2D ×2 (Y + UV)   ← stride-remove NV12 into cudaMalloc buffers
       │                                  (on Jetson iGPU this is coherence bookkeeping,
       │                                   not a real DMA copy — same LPDDR5 pages)
       │
       ├── NitrosImageBuilder::Build() ← wraps cuda_nv12_* in GXF VideoBuffer
       │   pub_{left,right}_nitros_   ← zero-copy GPU handle to downstream NITROS nodes
       │
       └── publish_visual_side ×2      ← BGRA→BGR (NEON), then:
               transport=compressed  → cv::imencode(".jpg") → CompressedImage
               transport=raw         → cv_bridge::CvImage  → Image
```

### Per-frame processing (CPU fallback path)

```
gst_buffer_map(system-memory)
       │
       ├── cv::Mat ROI split           ← left = combined[:, 0:ew],  right = combined[:, ew:]
       │
       ├── cv::cvtColor ×2             ← BGRx→BGR8 (NEON auto-vectorised at -O3)
       │
       └── publish_cpu_visual ×2       ← encoding conversion, optional resize, then:
               transport=compressed  → cv::imencode(".jpg") → CompressedImage
               transport=raw         → cv_bridge::CvImage  → Image
```

### Bandwidth comparison (1280×480 @ 30 fps, per eye)

| Stream | Bandwidth | Notes |
|---|---|---|
| NV12 NITROS | ~4.7 MB/s | GPU-resident, zero host copy |
| visual_stream (compressed, q=80) | ~0.5–1 MB/s | JPEG ~97% reduction vs raw |
| visual_stream (raw BGR) | ~27 MB/s | Full uncompressed, WiFi-hostile |

---

## Published Topics

All topic names are derived from `<prefix>/<suffix>` where the prefix is set by `topics.topic_name_prefix` in `params.yaml` (default: `/arducam/left` and `/arducam/right`).

| Topic | Type | Condition |
|---|---|---|
| `<prefix>/image/compressed` | `sensor_msgs/CompressedImage` | `visual_stream.enable=true` and `transport=compressed` |
| `<prefix>/image_raw` | `sensor_msgs/Image` | `visual_stream.enable=true` and `transport=raw` |
| `<prefix>/camera_info` | `sensor_msgs/CameraInfo` | always |
| `<prefix>/nitros_image_nv12` | `NitrosImage` | `nitros_image.enable=true`, `HAVE_NVBUF` compiled |

---

## TF Frames

One static transform is broadcast per camera at startup:

```
<extrinsics.relative_to>  ──►  <frame_id>
     (e.g. base_link)               (e.g. left_camera)
```

Rotation is specified as `[roll, pitch, yaw]` in degrees; translation as `[x, y, z]` in metres.

---

## Parameters

All parameters live under `arducam_dual_cam_node/ros__parameters` in `config/params.yaml`.

### Global

| Parameter | Type | Default | Description |
|---|---|---|---|
| `device` | string | `/dev/video0` | V4L2 device node |
| `width` | int | `1280` | Combined side-by-side width (pixels) |
| `height` | int | `480` | Combined height (pixels) |
| `fps` | int | `30` | Capture frame rate (0 = driver-negotiated) |
| `pixel_format` | string | `UYVY` | V4L2 input format (`UYVY` or `NV16`) |

Available capture modes for the B0573:

| Combined resolution | Per-eye resolution |
|---|---|
| 3840 × 1200 | 1920 × 1200 |
| 2560 × 720 | 1280 × 720 |
| **1280 × 480** | **640 × 480** (default) |

### Per-camera (`left_camera.*` / `right_camera.*`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `frame_id` | string | `left_camera` / `right_camera` | TF frame ID stamped on all messages |
| `topics.topic_name_prefix` | string | `/arducam/left` / `/arducam/right` | Prefix for all derived topic names |

#### `topics.visual_stream`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | bool | `true` | Publish this stream |
| `transport` | string | `compressed` | `compressed` = JPEG `sensor_msgs/CompressedImage`; `raw` = `sensor_msgs/Image` |
| `encoding` | string | `bgr8` | Pixel encoding: `bgr8` or `rgb8` |
| `jpeg_quality` | int | `80` | JPEG quality 0–100 (only used when `transport=compressed`) |
| `qos.reliability` | string | `best_effort` | `reliable` or `best_effort` |
| `qos.durability` | string | `volatile` | `volatile` or `transient_local` |
| `resolution.width` | int | `-1` | Output width (-1 = same as capture) |
| `resolution.height` | int | `-1` | Output height (-1 = same as capture) |

#### `topics.nitros_image`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | bool | `true` | Publish GPU-resident NitrosImage (requires `HAVE_NVBUF`) |
| `format` | string | `nv12` | `nv12`, `nv24`, `rgb8`, or `bgr8` |
| `qos.reliability` | string | `best_effort` | `reliable` or `best_effort` |
| `qos.durability` | string | `volatile` | `volatile` or `transient_local` |
| `resolution.width` | int | `-1` | Output width (-1 = same as capture) |
| `resolution.height` | int | `-1` | Output height (-1 = same as capture) |

#### `topics.camera_info`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `qos.reliability` | string | `reliable` | `reliable` or `best_effort` |
| `qos.durability` | string | `volatile` | `volatile` or `transient_local` |

#### `extrinsics`

| Parameter | Type | Description |
|---|---|---|
| `relative_to` | string | Parent TF frame (e.g. `base_link`) |
| `rotation` | double[3] | `[roll, pitch, yaw]` in degrees |
| `translation` | double[3] | `[x, y, z]` in metres |

#### `intrinsics`

| Parameter | Type | Description |
|---|---|---|
| `fx`, `fy`, `cx`, `cy` | double | Focal lengths and principal point (pixels) |
| `distortion_model` | string | `plumb_bob`, `rational_polynomial`, or `thin_prism_fisheye` |
| `distortion_coefficients` | double[] | D vector (5 or more coefficients) |
| `reflection_matrix.data` | double[9] | Rectification matrix R (row-major) |
| `projection_matrix.data` | double[12] | Projection matrix P (row-major, 3×4) |

---

## Quick Start

```bash
# Build
cd ~/ros2_ws
colcon build --packages-select arducam_dual_camera \
             --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Launch
source install/setup.bash
ros2 launch arducam_dual_camera dual_camera.launch.py

# Verify topics
ros2 topic list | grep arducam
ros2 topic hz /arducam/left/image/compressed
```

### View compressed stream in rviz2 (remote machine)

```bash
# Install the compressed image transport plugin if not present
sudo apt install ros-humble-image-transport-plugins

# In rviz2 → Add → By topic → /arducam/left/image/compressed → Image
```

### Switch to raw transport (local display, no compression)

Edit `config/params.yaml`:
```yaml
left_camera:
  topics:
    visual_stream:
      transport: "raw"
```
Then rebuild and relaunch.

---

## Dependencies

| Dependency | Required | Notes |
|---|---|---|
| ROS 2 Humble | Yes | |
| OpenCV 4 | Yes | `cv_bridge` |
| GStreamer 1.x | Yes | `gst-plugins-good`, `gst-plugins-bad` |
| NVIDIA JetPack 6.2+ | For HAVE_NVBUF | `nvbufsurface`, `nvbufsurftransform` |
| Isaac ROS NITROS | For HAVE_NVBUF | `isaac_ros_nitros`, `isaac_ros_nitros_image_type` |
| `tf2_ros` | Yes | static TF broadcaster |

---

## Package Layout

```
arducam_dual_camera/
├── config/
│   ├── params.yaml                 # all node parameters
│   └── nitros_context_graph.yaml   # minimal GXF graph for NitrosContext init
├── include/arducam_dual_camera/
│   └── arducam_dual_cam_node.hpp
├── launch/
│   ├── dual_camera.launch.py       # main launch file
│   └── list_modes.launch.py        # helper: list V4L2 camera modes
├── scripts/
│   └── list_camera_modes.py
└── src/
    ├── arducam_dual_cam_node.cpp
    └── main.cpp
```










## TODO:

- [ ] nitros `rgb8` and `bgr8` format support (currently only `nv12` and `nv24` is supported for nitros output)
- [ ] Add launch file with dynamic reconfigure support for parameters
