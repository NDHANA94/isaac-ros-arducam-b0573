/**
 * arducam_dual_cam_node.cpp
 *
 * Single ROS 2 composable node for the Arducam B0573 (GMSL2-to-CSI2) dual camera.
 *
 * ═══════════════════════════════════════════════════
 * Hardware acceleration — Jetson Orin Nano (JetPack 6)
 * ═══════════════════════════════════════════════════
 *
 * VIC (Video Image Compositor) — always used
 *   nvvidconv GStreamer element drives the VIC for all colour-space conversions
 *   (UYVY→NV12 preferred; UYVY→BGRx as fallback).
 *
 * NvBufSurface / NvBufSurfTransform  (HAVE_NVBUF path, preferred)
 *   When the pipeline negotiates memory:NVMM caps, gst_buffer_map() returns
 *   an NvBufSurface* — no pixel data is copied to CPU.  Two async VIC
 *   NvBufSurfTransformAsync() calls then crop the combined frame into left/right
 *   NV12 per-eye NvBufSurfaces entirely in VIC hardware.
 *
 * Isaac ROS NITROS  (HAVE_NVBUF path)
 *   NitrosImageBuilder::WithGpuData(cuda_ptr)->Build() wraps the per-eye
 *   NvBufSurface CUDA device pointer in a GXF VideoBuffer and publishes it.
 *   A co-located downstream NITROS node (e.g. isaac_ros_image_proc)
 *   receives the GXF handle with ZERO host DMA — the NV12 frame stays
 *   resident in GPU LPDDR5 from capture all the way through to inference.
 *
 *   Pipeline bandwidth comparison at 2560×720 @ 30 fps:
 *     NV12 NITROS:  2.77 MB/frame = 83 MB/s  (GPU only, zero host copy)  ← preferred
 *     BGRx CPU:     7.37 MB/frame = 221 MB/s on LPDDR5                   ← fallback
 *
 *   NITROS path CPU cost:
 *     NvBufSurfTransformAsync() housekeeping (µs)
 *     NitrosImageBuilder::Build()             (GXF entity alloc, µs)
 *     ZERO cv_bridge / cv::cvtColor / memcpy
 *
 * CPU fallback (HAVE_NVBUF not defined, or NVMM not negotiated)
 *   v4l2src BGRx + gst_buffer_map copy to CPU + cv::cvtColor × 2.
 *   Compiler is built with -O3 -march=armv8.2-a+simd+fp16+dotprod so NEON
 *   is auto-vectorised for the colour-conversion kernels.
 *
 * NVDEC — not applicable (raw sensor frames are uncompressed).
 *
 * ═══════════════════════════════
 * Pipeline fallback order
 * ═══════════════════════════════
 *   1. v4l2src(NVMM,UYVY) → nvvidconv/VIC → NV12(NVMM)  → appsink  [preferred]
 *   2. v4l2src(NVMM,UYVY) → nvvidconv/VIC → BGRx(NVMM)  → appsink
 *   3. v4l2src(UYVY)      → nvvidconv/VIC → BGRx         → appsink  [CPU path]
 *
 * Topics published (matches isaac_ros_argus_camera stereo pattern):
 *   /arducam/left/image_raw              sensor_msgs/Image   bgr8, always (rviz2)
 *   /arducam/right/image_raw             sensor_msgs/Image   bgr8, always (rviz2)
 *   /arducam/left/camera_info            sensor_msgs/CameraInfo, always
 *   /arducam/right/camera_info           sensor_msgs/CameraInfo, always
 *   /arducam/left/nitros_image_nv12      NitrosImage  NV12 GPU-resident, HAVE_NVBUF only
 *   /arducam/right/nitros_image_nv12     NitrosImage  NV12 GPU-resident, HAVE_NVBUF only
 *
 * Parameters:
 *   device              (string)  /dev/video0
 *   width               (int)     2560
 *   height              (int)     720
 *   fps                 (int)     30
 *   pixel_format        (string)  UYVY
 *
 *   cam_left / cam_right  (nested, mirrors params.yaml structure)
 *     .topic_name                     string   /arducam/{left,right}/image_raw
 *     .topic_qos.reliability          string   reliable | best_effort
 *     .topic_qos.durability           string   volatile | transient_local
 *     .output_resolution.{width,height} int    -1 = same as capture resolution
 *     .frame_id                       string   left_camera / right_camera
 *     .extrinsics.relative_to         string   base_link
 *     .extrinsics.rotation            double[] [roll, pitch, yaw] deg
 *     .extrinsics.translation         double[] [x, y, z] m
 *     .intrinsics.fx / fy / cx / cy   double
 *     .intrinsics.distortion_model    string   plumb_bob
 *     .intrinsics.distortion_coefficients double[]
 *     .intrinsics.reflection_matrix.data  double[9]   rectification R
 *     .intrinsics.projection_matrix.data  double[12]  projection P
 */

#include "arducam_dual_camera/arducam_dual_cam_node.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// NITROS: used for GPU-resident NitrosImage publishing in the HAVE_NVBUF path.
// sensor_msgs/Image publishers remain for the CPU fallback path.
#ifdef HAVE_NVBUF
#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
using NitrosImageBuilder = nvidia::isaac_ros::nitros::NitrosImageBuilder;
using nvidia::isaac_ros::nitros::NitrosImage;
#include "gxf/core/gxf.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#endif

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <rclcpp_components/register_node_macro.hpp>

namespace arducam_dual_camera
{

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
ArducamDualCamNode::ArducamDualCamNode(const rclcpp::NodeOptions & options)
: Node("arducam_dual_cam_node", options)
{
  // ── Global parameters ─────────────────────────────────────────────────────
  declare_parameter("device",        "/dev/video0");
  declare_parameter("width",         2560);
  declare_parameter("height",        720);
  declare_parameter("fps",           30);
  declare_parameter("pixel_format",  "UYVY");

  // ── Per-camera parameters (cam_left / cam_right) ───────────────────────────
  for (const std::string side : {"cam_left", "cam_right"}) {
    const bool is_left = (side == "cam_left");
    declare_parameter(side + ".topic_name_prefix",
      is_left ? "/arducam/left" : "/arducam/right");
    declare_parameter(side + ".publish_image_raw",  true);
    declare_parameter(side + ".nitros_format",       std::string("nv12"));
    declare_parameter(side + ".raw_qos.reliability",             std::string("reliable"));
    declare_parameter(side + ".raw_qos.durability",              std::string("volatile"));
    declare_parameter(side + ".info_qos.reliability",            std::string("reliable"));
    declare_parameter(side + ".info_qos.durability",             std::string("volatile"));
    declare_parameter(side + ".nitros_qos.reliability",          std::string("best_effort"));
    declare_parameter(side + ".nitros_qos.durability",           std::string("volatile"));
    declare_parameter(side + ".output_resolution.width",            -1);
    declare_parameter(side + ".output_resolution.height",           -1);
    declare_parameter(side + ".frame_id",
      is_left ? "left_camera" : "right_camera");
    declare_parameter(side + ".extrinsics.relative_to",             "base_link");
    declare_parameter(side + ".extrinsics.rotation",    std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter(side + ".extrinsics.translation", std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter(side + ".intrinsics.fx",                      900.0);
    declare_parameter(side + ".intrinsics.fy",                      900.0);
    declare_parameter(side + ".intrinsics.cx",                      640.0);
    declare_parameter(side + ".intrinsics.cy",                      360.0);
    declare_parameter(side + ".intrinsics.distortion_model",        "plumb_bob");
    declare_parameter(side + ".intrinsics.distortion_coefficients",
      std::vector<double>(5, 0.0));
    declare_parameter(side + ".intrinsics.reflection_matrix.data",
      std::vector<double>{1,0,0, 0,1,0, 0,0,1});
    declare_parameter(side + ".intrinsics.projection_matrix.data",
      std::vector<double>{1,0,0,0, 0,1,0,0, 0,0,1,0});
  }

  device_          = get_parameter("device").as_string();
  combined_width_  = get_parameter("width").as_int();
  combined_height_ = get_parameter("height").as_int();
  fps_             = get_parameter("fps").as_int();
  pixel_format_    = get_parameter("pixel_format").as_string();

  // Build full topic names from the prefix: <prefix>/image_raw, <prefix>/camera_info
  auto build_topic = [](const std::string & prefix, const std::string & suffix) {
    return prefix + "/" + suffix;
  };

  const std::string prefix_left  = get_parameter("cam_left.topic_name_prefix").as_string();
  const std::string prefix_right = get_parameter("cam_right.topic_name_prefix").as_string();
  topic_left_       = build_topic(prefix_left,  "image_raw");
  topic_right_      = build_topic(prefix_right, "image_raw");
  topic_left_info_  = build_topic(prefix_left,  "camera_info");
  topic_right_info_ = build_topic(prefix_right, "camera_info");
  frame_id_left_    = get_parameter("cam_left.frame_id").as_string();
  frame_id_right_   = get_parameter("cam_right.frame_id").as_string();
  out_w_left_       = get_parameter("cam_left.output_resolution.width").as_int();
  out_h_left_       = get_parameter("cam_left.output_resolution.height").as_int();
  out_w_right_      = get_parameter("cam_right.output_resolution.width").as_int();
  out_h_right_      = get_parameter("cam_right.output_resolution.height").as_int();
  qos_raw_rel_left_     = get_parameter("cam_left.raw_qos.reliability").as_string();
  qos_raw_dur_left_     = get_parameter("cam_left.raw_qos.durability").as_string();
  qos_raw_rel_right_    = get_parameter("cam_right.raw_qos.reliability").as_string();
  qos_raw_dur_right_    = get_parameter("cam_right.raw_qos.durability").as_string();
  qos_info_rel_left_    = get_parameter("cam_left.info_qos.reliability").as_string();
  qos_info_dur_left_    = get_parameter("cam_left.info_qos.durability").as_string();
  qos_info_rel_right_   = get_parameter("cam_right.info_qos.reliability").as_string();
  qos_info_dur_right_   = get_parameter("cam_right.info_qos.durability").as_string();
  qos_nitros_rel_left_  = get_parameter("cam_left.nitros_qos.reliability").as_string();
  qos_nitros_dur_left_  = get_parameter("cam_left.nitros_qos.durability").as_string();
  qos_nitros_rel_right_ = get_parameter("cam_right.nitros_qos.reliability").as_string();
  qos_nitros_dur_right_ = get_parameter("cam_right.nitros_qos.durability").as_string();
  pub_raw_left_     = get_parameter("cam_left.publish_image_raw").as_bool();
  pub_raw_right_    = get_parameter("cam_right.publish_image_raw").as_bool();
  nitros_fmt_left_  = get_parameter("cam_left.nitros_format").as_string();
  nitros_fmt_right_ = get_parameter("cam_right.nitros_format").as_string();

  RCLCPP_INFO(get_logger(),
    "Arducam B0573 | device=%s  combined=%dx%d  eye=%dx%d  fps=%d  fmt=%s",
    device_.c_str(), combined_width_, combined_height_,
    eye_width(), combined_height_, fps_, pixel_format_.c_str());
  RCLCPP_INFO(get_logger(),
    "Left  | prefix=%s  frame=%s  out=%dx%d  raw=%s  nitros_fmt=%s",
    prefix_left.c_str(), frame_id_left_.c_str(),
    eff_out_w(true), eff_out_h(true),
    pub_raw_left_  ? "yes" : "no", nitros_fmt_left_.c_str());
  RCLCPP_INFO(get_logger(),
    "        raw_qos=[%s/%s]  info_qos=[%s/%s]  nitros_qos=[%s/%s]",
    qos_raw_rel_left_.c_str(),    qos_raw_dur_left_.c_str(),
    qos_info_rel_left_.c_str(),   qos_info_dur_left_.c_str(),
    qos_nitros_rel_left_.c_str(), qos_nitros_dur_left_.c_str());
  RCLCPP_INFO(get_logger(),
    "Right | prefix=%s  frame=%s  out=%dx%d  raw=%s  nitros_fmt=%s",
    prefix_right.c_str(), frame_id_right_.c_str(),
    eff_out_w(false), eff_out_h(false),
    pub_raw_right_ ? "yes" : "no", nitros_fmt_right_.c_str());
  RCLCPP_INFO(get_logger(),
    "        raw_qos=[%s/%s]  info_qos=[%s/%s]  nitros_qos=[%s/%s]",
    qos_raw_rel_right_.c_str(),    qos_raw_dur_right_.c_str(),
    qos_info_rel_right_.c_str(),   qos_info_dur_right_.c_str(),
    qos_nitros_rel_right_.c_str(), qos_nitros_dur_right_.c_str());

  // ── Publishers (per-camera QoS) ─────────────────────────────────────────
  auto make_qos = [](const std::string & rel, const std::string & dur) {
    rclcpp::QoS q(rclcpp::KeepLast(10));
    (rel == "reliable") ? q.reliable()    : q.best_effort();
    (dur == "transient_local") ? q.transient_local() : q.durability_volatile();
    return q;
  };

  // ── Raw sensor_msgs/Image publishers — created only when publish_image_raw=true ──
  // topic = <prefix>/image_raw.  nullptr when disabled.
  if (pub_raw_left_) {
    pub_left_raw_  = create_publisher<sensor_msgs::msg::Image>(
      topic_left_,  make_qos(qos_raw_rel_left_,  qos_raw_dur_left_));
    RCLCPP_INFO(get_logger(), "image_raw LEFT  → %s", topic_left_.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "image_raw LEFT   disabled (publish_image_raw=false)");
  }
  if (pub_raw_right_) {
    pub_right_raw_ = create_publisher<sensor_msgs::msg::Image>(
      topic_right_, make_qos(qos_raw_rel_right_, qos_raw_dur_right_));
    RCLCPP_INFO(get_logger(), "image_raw RIGHT → %s", topic_right_.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "image_raw RIGHT  disabled (publish_image_raw=false)");
  }
  // camera_info is always published regardless of publish_image_raw
  pub_left_info_  = create_publisher<sensor_msgs::msg::CameraInfo>(
    topic_left_info_,  make_qos(qos_info_rel_left_,  qos_info_dur_left_));
  pub_right_info_ = create_publisher<sensor_msgs::msg::CameraInfo>(
    topic_right_info_, make_qos(qos_info_rel_right_, qos_info_dur_right_));

  // ── NITROS publishers (GPU-resident zero-copy, HAVE_NVBUF only) ────────────────
  // Topic: <prefix>/nitros_image_<format>  (cam_{left,right}.nitros_format)
  // Supported NitrosImageBuilder encodings: "nv12", "rgb8", "bgr8"
#ifdef HAVE_NVBUF
  topic_left_nitros_  = build_topic(prefix_left,  "nitros_image_" + nitros_fmt_left_);
  topic_right_nitros_ = build_topic(prefix_right, "nitros_image_" + nitros_fmt_right_);
  pub_left_nitros_  = create_publisher<NitrosImage>(
    topic_left_nitros_,  make_qos(qos_nitros_rel_left_,  qos_nitros_dur_left_));
  pub_right_nitros_ = create_publisher<NitrosImage>(
    topic_right_nitros_, make_qos(qos_nitros_rel_right_, qos_nitros_dur_right_));
  RCLCPP_INFO(get_logger(), "NITROS LEFT  → %s  (encoding=%s)",
    topic_left_nitros_.c_str(),  nitros_fmt_left_.c_str());
  RCLCPP_INFO(get_logger(), "NITROS RIGHT → %s  (encoding=%s)",
    topic_right_nitros_.c_str(), nitros_fmt_right_.c_str());
#endif

  // ── Build CameraInfo from inline intrinsic parameters ────────────────────
  cam_info_left_  = build_cam_info("cam_left",  eye_width(), combined_height_);
  cam_info_right_ = build_cam_info("cam_right", eye_width(), combined_height_);

  // ── Broadcast static TF transforms (extrinsics) ──────────────────────────
  tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
  broadcast_static_tf("cam_left",  frame_id_left_);
  broadcast_static_tf("cam_right", frame_id_right_);

  // ── GStreamer ────────────────────────────────────────────────────────────
  int   argc = 0;
  char ** argv = nullptr;
  gst_init(&argc, &argv);

#ifdef HAVE_NVBUF
  init_nitros_context();   // load cuda + multimedia GXF extensions before first Build()
#endif

  build_pipeline();   // also calls init_nvbuf_surfaces() when HAVE_NVBUF

  running_        = true;
  capture_thread_ = std::thread(&ArducamDualCamNode::capture_loop, this);
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
ArducamDualCamNode::~ArducamDualCamNode()
{
  running_ = false;

  // Send EOS on v4l2src to unblock the blocking pull_sample()
  if (pipeline_) {
    GstElement * src = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
    if (src) {
      gst_element_send_event(src, gst_event_new_eos());
      gst_object_unref(src);
    }
  }

  if (capture_thread_.joinable()) capture_thread_.join();

  if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_NULL);

  // gst_bin_get_by_name() adds a reference — release before the pipeline
  if (appsink_) { gst_object_unref(appsink_); appsink_ = nullptr; }
  if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }

#ifdef HAVE_NVBUF
  cleanup_nvbuf_surfaces();
  nitros_ctx_.destroy();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// build_cam_info()
//   Reads the inline intrinsic parameters for the given side ("cam_left" or
//   "cam_right") and constructs a sensor_msgs::CameraInfo ready for publishing.
//   The header (stamp / frame_id) is intentionally left empty here; it is filled
//   in at publish time so that each message carries the correct per-frame stamp.
// ─────────────────────────────────────────────────────────────────────────────
sensor_msgs::msg::CameraInfo ArducamDualCamNode::build_cam_info(
  const std::string & side, int w, int h)
{
  sensor_msgs::msg::CameraInfo ci;
  ci.width  = static_cast<uint32_t>(w);
  ci.height = static_cast<uint32_t>(h);

  ci.distortion_model =
    get_parameter(side + ".intrinsics.distortion_model").as_string();

  auto d = get_parameter(side + ".intrinsics.distortion_coefficients").as_double_array();
  ci.d.assign(d.begin(), d.end());

  const double fx = get_parameter(side + ".intrinsics.fx").as_double();
  const double fy = get_parameter(side + ".intrinsics.fy").as_double();
  const double cx = get_parameter(side + ".intrinsics.cx").as_double();
  const double cy = get_parameter(side + ".intrinsics.cy").as_double();

  // Camera matrix K (3×3, row-major)
  ci.k = {fx,  0.0, cx,
          0.0, fy,  cy,
          0.0, 0.0, 1.0};

  // Rectification matrix R (3×3, row-major)
  auto r_data = get_parameter(side + ".intrinsics.reflection_matrix.data").as_double_array();
  if (r_data.size() == 9) {
    std::copy(r_data.begin(), r_data.end(), ci.r.begin());
  } else {
    ci.r = {1, 0, 0,  0, 1, 0,  0, 0, 1};
  }

  // Projection matrix P (3×4, row-major)
  auto p_data = get_parameter(side + ".intrinsics.projection_matrix.data").as_double_array();
  if (p_data.size() == 12) {
    std::copy(p_data.begin(), p_data.end(), ci.p.begin());
  } else {
    ci.p = {fx, 0.0, cx, 0.0,
            0.0, fy,  cy, 0.0,
            0.0, 0.0, 1.0, 0.0};
  }

  RCLCPP_INFO(get_logger(),
    "CameraInfo built for %s: %dx%d  fx=%.1f fy=%.1f cx=%.1f cy=%.1f  D[%zu]",
    side.c_str(), w, h, fx, fy, cx, cy, ci.d.size());
  return ci;
}

// ─────────────────────────────────────────────────────────────────────────────
// broadcast_static_tf()
//   Reads the extrinsics for the given side and publishes a latched static TF
//   transform from the parent frame (extrinsics.relative_to) to child_frame.
//   rotation is [roll, pitch, yaw] in degrees; translation is [x, y, z] in m.
// ─────────────────────────────────────────────────────────────────────────────
void ArducamDualCamNode::broadcast_static_tf(
  const std::string & side, const std::string & child_frame)
{
  const std::string parent =
    get_parameter(side + ".extrinsics.relative_to").as_string();
  auto rpy = get_parameter(side + ".extrinsics.rotation").as_double_array();
  auto xyz = get_parameter(side + ".extrinsics.translation").as_double_array();

  constexpr double DEG2RAD = M_PI / 180.0;
  tf2::Quaternion q;
  q.setRPY(
    (rpy.size() > 0 ? rpy[0] : 0.0) * DEG2RAD,
    (rpy.size() > 1 ? rpy[1] : 0.0) * DEG2RAD,
    (rpy.size() > 2 ? rpy[2] : 0.0) * DEG2RAD);
  q.normalize();

  geometry_msgs::msg::TransformStamped t{};
  t.header.stamp    = now();
  t.header.frame_id = parent;
  t.child_frame_id  = child_frame;
  t.transform.translation.x = xyz.size() > 0 ? xyz[0] : 0.0;
  t.transform.translation.y = xyz.size() > 1 ? xyz[1] : 0.0;
  t.transform.translation.z = xyz.size() > 2 ? xyz[2] : 0.0;
  t.transform.rotation.x = q.x();
  t.transform.rotation.y = q.y();
  t.transform.rotation.z = q.z();
  t.transform.rotation.w = q.w();

  tf_static_broadcaster_->sendTransform(t);

  RCLCPP_INFO(get_logger(),
    "Static TF: %s → %s  t=[%.3f,%.3f,%.3f]  rpy_deg=[%.1f,%.1f,%.1f]",
    parent.c_str(), child_frame.c_str(),
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
    rpy.size() > 0 ? rpy[0] : 0.0,
    rpy.size() > 1 ? rpy[1] : 0.0,
    rpy.size() > 2 ? rpy[2] : 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// try_build_pipeline()
//   nvmm    – request memory:NVMM caps (DMA-BUF path, avoids CPU pixel copy)
//   out_fmt – nvvidconv output colour format: "NV12" (preferred, 1.5 B/px) or
//             "BGRx" (4 B/px, used for CPU fallback path)
//
// NV12 (YUV 4:2:0 semi-planar) costs only 1.5 bytes/pixel vs 4 bytes/pixel for
// BGRx, reducing VIC→LPDDR5 write bandwidth by 2.7×.  When HAVE_NVBUF is
// compiled in, NvBufSurfTransform performs the NV12→BGR conversion + crop via
// VIC, so the CPU never sees the combined-format frame at all.
// ─────────────────────────────────────────────────────────────────────────────
bool ArducamDualCamNode::try_build_pipeline(bool nvmm, const std::string & out_fmt)
{
  // v4l2src always outputs system-memory buffers (the tegra-video V4L2 driver
  // does not support memory:NVMM on the source pad for UYVY).
  // memory:NVMM is requested only on nvvidconv's OUTPUT so that the VIC DMA-writes
  // the converted frame directly into an NvBufSurface without a CPU bounce copy.
  // gst_buffer_map() on an NVMM appsink buffer returns NvBufSurface* (a small
  // struct, not pixel data) — this is what allows the zero-copy VIC path.
  const std::string dst_mem = nvmm ? "(memory:NVMM)" : "";

  const std::string fps_caps = (fps_ > 0)
    ? (std::string(",framerate=") + std::to_string(fps_) + "/1")
    : "";

  const std::string pipeline_str =
    "v4l2src name=src device=" + device_ +
    " ! video/x-raw,format=" + pixel_format_ +
    ",width="  + std::to_string(combined_width_) +
    ",height=" + std::to_string(combined_height_) +
    fps_caps +
    " ! nvvidconv"
    " ! video/x-raw" + dst_mem + ",format=" + out_fmt +
    " ! appsink name=sink sync=false max-buffers=3 drop=true";

  RCLCPP_INFO(get_logger(), "Trying pipeline (%s/%s): %s",
    nvmm ? "NVMM" : "system-memory", out_fmt.c_str(), pipeline_str.c_str());

  GError * error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
  if (!pipeline_ || error) {
    RCLCPP_WARN(get_logger(), "gst_parse_launch failed: %s",
      error ? error->message : "(unknown)");
    if (error)     g_error_free(error);
    if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
    return false;
  }

  appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!appsink_) {
    RCLCPP_WARN(get_logger(), "Could not find appsink element");
    gst_object_unref(pipeline_); pipeline_ = nullptr;
    return false;
  }

  gst_app_sink_set_emit_signals(GST_APP_SINK(appsink_), FALSE);

  GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    RCLCPP_WARN(get_logger(), "Pipeline failed immediately on PLAYING transition");
    gst_object_unref(appsink_); appsink_ = nullptr;
    gst_object_unref(pipeline_); pipeline_ = nullptr;
    return false;
  }

  if (ret == GST_STATE_CHANGE_ASYNC) {
    GstState state = GST_STATE_NULL;
    ret = gst_element_get_state(pipeline_, &state, nullptr,
                                static_cast<GstClockTime>(5) * GST_SECOND);
    if (ret == GST_STATE_CHANGE_FAILURE || state != GST_STATE_PLAYING) {
      RCLCPP_WARN(get_logger(),
        "Pipeline did not reach PLAYING within 5 s (ret=%d state=%d)",
        static_cast<int>(ret), static_cast<int>(state));
      gst_element_set_state(pipeline_, GST_STATE_NULL);
      gst_object_unref(appsink_); appsink_ = nullptr;
      gst_object_unref(pipeline_); pipeline_ = nullptr;
      return false;
    }
  }

  use_nvmm_    = nvmm;
  gst_out_fmt_ = out_fmt;
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_pipeline()
//   Tries three pipeline variants in order of preference:
//     1. NVMM + NV12  (1.5 B/px, 2.7× less LPDDR5 pressure than BGRx)
//     2. NVMM + BGRx  (4 B/px, but VIC path still avoids CPU pixel copy)
//     3. System + BGRx (no NVMM; CPU gst_buffer_map + cv::cvtColor fallback)
//
//   NVMM variants are only attempted when HAVE_NVBUF is compiled in, because
//   gst_buffer_map() on an NVMM appsink returns NvBufSurface* (not pixel data).
//   Without NvBufSurface the CPU path cannot interpret that pointer.
// ─────────────────────────────────────────────────────────────────────────────
void ArducamDualCamNode::build_pipeline()
{
  bool ok = false;

#ifdef HAVE_NVBUF
  // ── NVMM paths (preferred) ───────────────────────────────────────────────
  // NV12 output saves 2.7× DMA bandwidth vs BGRx; VIC handles the
  // NV12→BGR format conversion inside NvBufSurfTransformAsync.
  ok = try_build_pipeline(/*nvmm=*/true, "NV12");
  if (!ok) {
    RCLCPP_WARN(get_logger(),
      "NVMM+NV12 unavailable — trying NVMM+BGRx");
    ok = try_build_pipeline(/*nvmm=*/true, "BGRx");
  }

  if (ok) {
    // Try to allocate pre-mapped BGR destination surfaces for VIC transforms.
    init_nvbuf_surfaces();
    if (!use_nvbuf_) {
      // NvBufSurface init failed (e.g. on some early JetPack 6 kernels).
      // Tear down the NVMM pipeline and fall through to system-memory BGRx.
      RCLCPP_WARN(get_logger(),
        "NvBufSurface init failed — abandoning NVMM, rebuilding with "
        "system-memory BGRx");
      gst_element_set_state(pipeline_, GST_STATE_NULL);
      gst_object_unref(appsink_);  appsink_  = nullptr;
      gst_object_unref(pipeline_); pipeline_ = nullptr;
      ok = false;
    }
  }
#endif  // HAVE_NVBUF

  // ── System-memory fallback ───────────────────────────────────────────────
  if (!ok) {
    RCLCPP_WARN(get_logger(),
      "Using system-memory BGRx pipeline (CPU colour-conversion path)");
    ok = try_build_pipeline(/*nvmm=*/false, "BGRx");
  }

  if (!ok) {
    RCLCPP_FATAL(get_logger(), "All pipeline variants failed — shutting down");
    rclcpp::shutdown();
    return;
  }

  RCLCPP_INFO(get_logger(), "Pipeline PLAYING  memory=%s  format=%s%s",
    use_nvmm_ ? "NVMM" : "system",
    gst_out_fmt_.c_str(),
#ifdef HAVE_NVBUF
    use_nvbuf_ ? "  [VIC NvBufSurfTransform path]" : "  [CPU path]"
#else
    "  [CPU path — HAVE_NVBUF not compiled]"
#endif
  );

  // ── CLOCK_MONOTONIC → CLOCK_REALTIME offset ───────────────────────────────
  {
    struct timespec mono{}, real{};
    clock_gettime(CLOCK_MONOTONIC, &mono);
    clock_gettime(CLOCK_REALTIME,  &real);
    gst_clock_offset_ =
      (static_cast<int64_t>(real.tv_sec)  - static_cast<int64_t>(mono.tv_sec))  * 1'000'000'000LL
    + (static_cast<int64_t>(real.tv_nsec) - static_cast<int64_t>(mono.tv_nsec));
    RCLCPP_INFO(get_logger(),
      "Clock offset REALTIME−MONOTONIC: %.3f s",
      static_cast<double>(gst_clock_offset_) * 1e-9);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// NvBufSurface VIC path  (compiled only when HAVE_NVBUF is defined)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef HAVE_NVBUF

// ── init_nitros_context() ─────────────────────────────────────────────────────
// Pre-registers nvidia::gxf::VideoBuffer (and cuda types) in the shared GXF
// context so NitrosImageBuilder::Build() can allocate GXF entities.
//
// NitrosImageBuilder::Build() lazily creates a NitrosContext and loads:
//   std / isaac_gxf_helpers / isaac_sight / isaac_atlas
// but does NOT load gxf_cuda or gxf_multimedia, so VideoBuffer is unknown
// and GXF panics.  Loading those two extensions here, before the first
// Build() call, fixes the panic — the shared GXF context is global, so
// extensions loaded via our nitros_ctx_ instance are visible to Build().
//
// Pattern mirrors isaac_ros_argus_camera:GetArgusNitrosContext().
void ArducamDualCamNode::init_nitros_context()
{
  const std::string gxf_dir =
    ament_index_cpp::get_package_share_directory("isaac_ros_gxf");

  // Load std first — registers nvidia::gxf::Allocator and System types that
  // gxf_cuda depends on.  Skipping this causes GXF_FACTORY_UNKNOWN_CLASS_NAME
  // when gxf_cuda tries to register its CudaAllocator component.
  gxf_result_t r = nitros_ctx_.loadExtension(gxf_dir, "gxf/lib/std/libgxf_std.so");
  if (r != GXF_SUCCESS) {
    RCLCPP_WARN(get_logger(),
      "init_nitros_context: loadExtension(gxf_std) returned %d", r);
  }

  // Load CUDA extension — prerequisite for gxf_multimedia's CUDA buffer types.
  r = nitros_ctx_.loadExtension(gxf_dir, "gxf/lib/cuda/libgxf_cuda.so");
  if (r != GXF_SUCCESS) {
    RCLCPP_WARN(get_logger(),
      "init_nitros_context: loadExtension(gxf_cuda) returned %d — VideoBuffer may panic", r);
  }

  // Load multimedia extension — registers nvidia::gxf::VideoBuffer.
  r = nitros_ctx_.loadExtension(gxf_dir, "gxf/lib/multimedia/libgxf_multimedia.so");
  if (r != GXF_SUCCESS) {
    RCLCPP_WARN(get_logger(),
      "init_nitros_context: loadExtension(gxf_multimedia) returned %d — VideoBuffer may panic", r);
  }

  // Load a minimal graph (empty — no schedulable entities); satisfies loadApplication
  // requirement before runGraphAsync().  "No GXF scheduler" warning is expected.
  const std::string pkg_dir =
    ament_index_cpp::get_package_share_directory("arducam_dual_camera");
  r = nitros_ctx_.loadApplication(pkg_dir + "/config/nitros_context_graph.yaml");
  if (r != GXF_SUCCESS) {
    RCLCPP_WARN(get_logger(),
      "init_nitros_context: loadApplication returned %d", r);
  }

  r = nitros_ctx_.runGraphAsync();
  if (r != GXF_SUCCESS) {
    RCLCPP_WARN(get_logger(),
      "init_nitros_context: runGraphAsync returned %d", r);
  }

  RCLCPP_INFO(get_logger(),
    "NITROS context: gxf_std + gxf_cuda + gxf_multimedia loaded (VideoBuffer registered)");
}

// ── init_nvbuf_surfaces() ─────────────────────────────────────────────────────
// Pre-allocates two NV12 SURFACE_ARRAY destination NvBufSurfaces (one per eye).
//
// Format/memory choice — NV12 + NVBUF_MEM_DEFAULT (= SURFACE_ARRAY on Orin):
//   • VIC (NvBufSurfTransform) REQUIRES SURFACE_ARRAY as the destination.
//     NVBUF_MEM_CUDA_DEVICE (raw CUDA heap) is NOT supported by the VIC HW
//     unit — it will return -1 "Surface type not supported for transformation".
//   • On Jetson Orin (iGPU), SURFACE_ARRAY buffers are allocated via nvmap
//     with IOVA mapping.  The same physical LPDDR5 pages are exposed to CUDA
//     as a device pointer in surfaceList[0].dataPtr, so
//     NitrosImageBuilder::WithGpuData(dataPtr) works zero-copy.
//   • NvBufSurfaceMap(planeIdx=-1, READ_WRITE) maps BOTH Y (addr[0]) and UV
//     (addr[1]) planes.  planeIdx=0 leaves addr[1] null; std::memcpy from null
//     in nv12_to_bgr_mat() would SIGSEGV immediately on the UV copy.
void ArducamDualCamNode::init_nvbuf_surfaces()
{
  NvBufSurfaceCreateParams params;
  memset(&params, 0, sizeof(params));
  params.gpuId       = 0;
  params.width       = static_cast<uint32_t>(eye_width());
  params.height      = static_cast<uint32_t>(combined_height_);
  // NV12: VIC crops NV12→NV12 natively (pure geometric, no colour conversion).
  params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
  params.layout      = NVBUF_LAYOUT_PITCH;
  // NVBUF_MEM_DEFAULT = SURFACE_ARRAY on Jetson Orin — required by VIC.
  // dataPtr is also a valid CUDA device pointer on Jetson (unified memory).
  params.memType     = NVBUF_MEM_DEFAULT;

  int ret_l = NvBufSurfaceCreate(&nvbuf_left_,  1, &params);
  int ret_r = NvBufSurfaceCreate(&nvbuf_right_, 1, &params);

  if (ret_l != 0 || ret_r != 0) {
    RCLCPP_WARN(get_logger(),
      "NvBufSurfaceCreate (NV12/SURFACE_ARRAY) failed (ret_l=%d ret_r=%d) — VIC path disabled",
      ret_l, ret_r);
    if (nvbuf_left_)  { NvBufSurfaceDestroy(nvbuf_left_);  nvbuf_left_  = nullptr; }
    if (nvbuf_right_) { NvBufSurfaceDestroy(nvbuf_right_); nvbuf_right_ = nullptr; }
    return;  // use_nvbuf_ stays false
  }

  // ── BGRA surfaces for image_raw (VIC NV12→BGRA hardware colour conversion) ──
  // VIC converts NV12→BGRA in the same crop pass — no CPU cvtColor or stride
  // memcpy needed.  mappedAddr.addr[0] is directly usable as a cv::Mat pointer.
  NvBufSurfaceCreateParams raw_params = params;
  raw_params.colorFormat = NVBUF_COLOR_FORMAT_BGRA;   // packed 4-channel, 1 plane

  int ret_rl = NvBufSurfaceCreate(&nvbuf_raw_left_,  1, &raw_params);
  int ret_rr = NvBufSurfaceCreate(&nvbuf_raw_right_, 1, &raw_params);

  if (ret_rl != 0 || ret_rr != 0) {
    RCLCPP_WARN(get_logger(),
      "NvBufSurfaceCreate (BGRA/SURFACE_ARRAY) failed (ret_l=%d ret_r=%d)"
      " — image_raw will use CPU cvtColor fallback", ret_rl, ret_rr);
    if (nvbuf_raw_left_)  { NvBufSurfaceDestroy(nvbuf_raw_left_);  nvbuf_raw_left_  = nullptr; }
    if (nvbuf_raw_right_) { NvBufSurfaceDestroy(nvbuf_raw_right_); nvbuf_raw_right_ = nullptr; }
    // Don't return — NV12 NITROS path still works; raw fallback handled in process_sample_nvbuf
  }

  // Map the SURFACE_ARRAY buffers for CPU access.
  // planeIdx=-1  → map ALL planes at once (Y = addr[0], UV = addr[1] for NV12).
  //   planeIdx=0 would leave addr[1] null → std::memcpy on the UV plane SEGFAULTs.
  // NVBUF_MAP_READ_WRITE → VIC writes to this surface as the transform destination;
  //   READ-only mapping does not reflect those writes after NvBufSurfaceSyncForCpu.
  if (NvBufSurfaceMap(nvbuf_left_,  0, -1, NVBUF_MAP_READ_WRITE) != 0 ||
      NvBufSurfaceMap(nvbuf_right_, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
    RCLCPP_WARN(get_logger(),
      "NvBufSurfaceMap (NV12) failed — VIC path disabled");
    NvBufSurfaceDestroy(nvbuf_left_);  nvbuf_left_  = nullptr;
    NvBufSurfaceDestroy(nvbuf_right_); nvbuf_right_ = nullptr;
    if (nvbuf_raw_left_)  { NvBufSurfaceDestroy(nvbuf_raw_left_);  nvbuf_raw_left_  = nullptr; }
    if (nvbuf_raw_right_) { NvBufSurfaceDestroy(nvbuf_raw_right_); nvbuf_raw_right_ = nullptr; }
    return;
  }

  // Map BGRA surfaces (single plane, planeIdx=-1 == 0 for packed formats, but
  // -1 is safe and consistent).
  if (nvbuf_raw_left_ && nvbuf_raw_right_) {
    if (NvBufSurfaceMap(nvbuf_raw_left_,  0, -1, NVBUF_MAP_READ_WRITE) != 0 ||
        NvBufSurfaceMap(nvbuf_raw_right_, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
      RCLCPP_WARN(get_logger(),
        "NvBufSurfaceMap (BGRA) failed — image_raw will use CPU cvtColor fallback");
      NvBufSurfaceDestroy(nvbuf_raw_left_);  nvbuf_raw_left_  = nullptr;
      NvBufSurfaceDestroy(nvbuf_raw_right_); nvbuf_raw_right_ = nullptr;
    }
  }

  // Allocate packed NV12 CUDA device buffers (Y plane + interleaved UV plane).
  // These are real cudaMalloc allocations whose pointers are valid CUDA device
  // addresses — required by VPI (called inside NitrosImageBuilder::Build()).
  // SURFACE_ARRAY dataPtr is a CPU nvmap virtual address that VPI rejects with
  // cudaErrorInvalidValue in cudaPointerAttributes(); cudaMalloc buffers pass.
  const size_t nv12_size = static_cast<size_t>(eye_width()) *
                           static_cast<size_t>(combined_height_) * 3 / 2;
  cudaError_t ce_l = cudaMalloc(&cuda_nv12_left_,  nv12_size);
  cudaError_t ce_r = cudaMalloc(&cuda_nv12_right_, nv12_size);
  if (ce_l != cudaSuccess || ce_r != cudaSuccess) {
    RCLCPP_WARN(get_logger(),
      "cudaMalloc NV12 staging buf failed (l=%d r=%d) — VIC path disabled",
      static_cast<int>(ce_l), static_cast<int>(ce_r));
    NvBufSurfaceUnMap(nvbuf_left_,  0, -1);
    NvBufSurfaceUnMap(nvbuf_right_, 0, -1);
    NvBufSurfaceDestroy(nvbuf_left_);  nvbuf_left_  = nullptr;
    NvBufSurfaceDestroy(nvbuf_right_); nvbuf_right_ = nullptr;
    if (cuda_nv12_left_)  { cudaFree(cuda_nv12_left_);  cuda_nv12_left_  = nullptr; }
    if (cuda_nv12_right_) { cudaFree(cuda_nv12_right_); cuda_nv12_right_ = nullptr; }
    return;
  }

  use_nvbuf_ = true;
  RCLCPP_INFO(get_logger(),
    "NvBufSurface ready  eye=%dx%d"
    "  NV12(NITROS): cuda_left=%p cuda_right=%p  buf=%zu B"
    "  BGRA(image_raw): %s",
    eye_width(), combined_height_,
    cuda_nv12_left_, cuda_nv12_right_, nv12_size,
    (nvbuf_raw_left_ && nvbuf_raw_right_) ? "VIC hw path" : "CPU cvtColor fallback");
}

// ── cleanup_nvbuf_surfaces() ──────────────────────────────────────────────────
void ArducamDualCamNode::cleanup_nvbuf_surfaces()
{
  if (nvbuf_left_) {
    NvBufSurfaceUnMap(nvbuf_left_,  0, -1);
    NvBufSurfaceDestroy(nvbuf_left_);
    nvbuf_left_ = nullptr;
  }
  if (nvbuf_right_) {
    NvBufSurfaceUnMap(nvbuf_right_, 0, -1);
    NvBufSurfaceDestroy(nvbuf_right_);
    nvbuf_right_ = nullptr;
  }
  if (nvbuf_raw_left_) {
    NvBufSurfaceUnMap(nvbuf_raw_left_,  0, -1);
    NvBufSurfaceDestroy(nvbuf_raw_left_);
    nvbuf_raw_left_ = nullptr;
  }
  if (nvbuf_raw_right_) {
    NvBufSurfaceUnMap(nvbuf_raw_right_, 0, -1);
    NvBufSurfaceDestroy(nvbuf_raw_right_);
    nvbuf_raw_right_ = nullptr;
  }
  if (cuda_nv12_left_)  { cudaFree(cuda_nv12_left_);  cuda_nv12_left_  = nullptr; }
  if (cuda_nv12_right_) { cudaFree(cuda_nv12_right_); cuda_nv12_right_ = nullptr; }
  use_nvbuf_ = false;
}

// ── process_sample_nvbuf() ────────────────────────────────────────────────────
// VIC + NITROS publishing path.
//
// Memory flow:
//   gst_buffer_map(NVMM) → NvBufSurface* src     [O(1), 64-byte struct]
//   NvBufSurfTransformAsync ×2  → VIC NV12 crop → nvbuf_left_ / nvbuf_right_
//     (SURFACE_ARRAY — the only dst type VIC accepts on Orin)
//   Wait sync objects → gst_buffer_unmap
//   NvBufSurfaceSyncForCpu ×2   → ensure CPU view is coherent
//   cudaMemcpy2D Y+UV planes    → cuda_nv12_left_ / cuda_nv12_right_
//     (SURFACE_ARRAY dataPtr is CPU nvmap virtual addr; VPI rejects it with
//      cudaErrorInvalidValue; we bridge to real cudaMalloc device memory)
//     On Jetson iGPU (unified LPDDR5) this is coherence-bookkeeping only.
//   NitrosImageBuilder::WithGpuData(cuda_nv12_*)->Build() → NitrosImage
//   pub_{left,right}_->publish(nitros_img)  → GXF VideoBuffer delivered
//
// Returns false if any VIC operation fails; caller rebuilds to CPU pipeline.
bool ArducamDualCamNode::process_sample_nvbuf(GstBuffer * buf, const rclcpp::Time & stamp)
{
  GstMapInfo map{};
  // NVMM gst_buffer_map returns the 64-byte NvBufSurface* struct — O(1).
  if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
    return false;
  }

  NvBufSurface * src = reinterpret_cast<NvBufSurface *>(map.data);
  const int ew = eye_width();

  // Crop rectangles: {top, left, width, height}
  NvBufSurfTransformRect rect_left  = {0, 0,
    static_cast<uint32_t>(ew), static_cast<uint32_t>(combined_height_)};
  NvBufSurfTransformRect rect_right = {0, static_cast<uint32_t>(ew),
    static_cast<uint32_t>(ew), static_cast<uint32_t>(combined_height_)};

  NvBufSurfTransformParams tp_left{};
  tp_left.transform_flag   = NVBUFSURF_TRANSFORM_CROP_SRC;
  tp_left.transform_filter = NvBufSurfTransformInter_Nearest;
  tp_left.src_rect         = &rect_left;

  NvBufSurfTransformParams tp_right{};
  tp_right.transform_flag   = NVBUFSURF_TRANSFORM_CROP_SRC;
  tp_right.transform_filter = NvBufSurfTransformInter_Nearest;
  tp_right.src_rect         = &rect_right;

  NvBufSurfTransformSyncObj_t sync_left  = nullptr;
  NvBufSurfTransformSyncObj_t sync_right = nullptr;

  auto err_l = NvBufSurfTransformAsync(src, nvbuf_left_,  &tp_left,  &sync_left);
  auto err_r = NvBufSurfTransformAsync(src, nvbuf_right_, &tp_right, &sync_right);

  // ── image_raw VIC path: NV12→BGRA crop in the same hardware pass ───────────
  // Kick off both BGRA transforms before waiting on ANY sync object so all four
  // VIC jobs run concurrently.  sync_raw_* are only used if nvbuf_raw_* allocated.
  const bool want_raw = pub_raw_left_ || pub_raw_right_;
  const bool have_raw_surfs = nvbuf_raw_left_ && nvbuf_raw_right_;
  NvBufSurfTransformSyncObj_t sync_raw_left  = nullptr;
  NvBufSurfTransformSyncObj_t sync_raw_right = nullptr;

  if (want_raw && have_raw_surfs) {
    // FILTER flag is required when color format differs between src and dst so
    // VIC applies the colour matrix in the same pass as the crop.
    NvBufSurfTransformParams tp_raw_left  = tp_left;
    NvBufSurfTransformParams tp_raw_right = tp_right;
    tp_raw_left.transform_flag  |= NVBUFSURF_TRANSFORM_FILTER;
    tp_raw_right.transform_flag |= NVBUFSURF_TRANSFORM_FILTER;
    NvBufSurfTransformAsync(src, nvbuf_raw_left_,  &tp_raw_left,  &sync_raw_left);
    NvBufSurfTransformAsync(src, nvbuf_raw_right_, &tp_raw_right, &sync_raw_right);
  }

  if (err_l != NvBufSurfTransformError_Success ||
      err_r != NvBufSurfTransformError_Success)
  {
    RCLCPP_WARN_ONCE(get_logger(),
      "NvBufSurfTransformAsync (NV12) failed (err_l=%d err_r=%d) — "
      "disabling VIC path", static_cast<int>(err_l), static_cast<int>(err_r));
    if (sync_left)       NvBufSurfTransformSyncObjDestroy(&sync_left);
    if (sync_right)      NvBufSurfTransformSyncObjDestroy(&sync_right);
    if (sync_raw_left)   NvBufSurfTransformSyncObjDestroy(&sync_raw_left);
    if (sync_raw_right)  NvBufSurfTransformSyncObjDestroy(&sync_raw_right);
    gst_buffer_unmap(buf, &map);
    use_nvbuf_ = false;
    return false;
  }

  // Wait for all VIC jobs — NV12 (NITROS) and BGRA (image_raw) run concurrently
  NvBufSurfTransformSyncObjWait(sync_left,  -1);
  NvBufSurfTransformSyncObjWait(sync_right, -1);
  NvBufSurfTransformSyncObjDestroy(&sync_left);
  NvBufSurfTransformSyncObjDestroy(&sync_right);
  if (sync_raw_left)  {
    NvBufSurfTransformSyncObjWait(sync_raw_left,  -1);
    NvBufSurfTransformSyncObjDestroy(&sync_raw_left);
  }
  if (sync_raw_right) {
    NvBufSurfTransformSyncObjWait(sync_raw_right, -1);
    NvBufSurfTransformSyncObjDestroy(&sync_raw_right);
  }

  gst_buffer_unmap(buf, &map);  // release NVMM src reference; dst is independent

  // ── CPU-sync then copy SURFACE_ARRAY → CUDA device buffers ─────────────────
  // SURFACE_ARRAY (nvmap) dataPtr is a CPU virtual address, NOT a CUDA device
  // pointer.  VPI calls cudaPointerAttributes() inside NitrosImageBuilder and
  // gets cudaErrorInvalidValue, crashing.  Bridge: sync CPU view with
  // NvBufSurfaceSyncForCpu, then use cudaMemcpy2D (Y + UV separately, because
  // SURFACE_ARRAY has padding per-plane) into pre-allocated cudaMalloc buffers.
  // On Jetson iGPU (unified LPDDR5) this is coherence-bookkeeping, not a real
  // DMA copy — the physical pages are the same.
  // Sync CPU view: NV12 surfaces (for cudaMemcpy2D) and BGRA surfaces (for cv::Mat wrap)
  NvBufSurfaceSyncForCpu(nvbuf_left_,  0, 0);
  NvBufSurfaceSyncForCpu(nvbuf_right_, 0, 0);
  if (have_raw_surfs && want_raw) {
    NvBufSurfaceSyncForCpu(nvbuf_raw_left_,  0, 0);
    NvBufSurfaceSyncForCpu(nvbuf_raw_right_, 0, 0);
  }

  // Copy NV12 planes (with stride removal) into packed cudaMalloc buffer.
  // dst layout: [Y: ew*h bytes][UV: ew*(h/2) bytes] — stride == ew (no padding).
  auto copy_nv12_to_cuda = [&](NvBufSurface * surf, void * cuda_dst, int ew, int h) {
    auto * y_src  = static_cast<const uint8_t *>(surf->surfaceList[0].mappedAddr.addr[0]);
    auto * uv_src = static_cast<const uint8_t *>(surf->surfaceList[0].mappedAddr.addr[1]);
    const size_t spitch_y  = surf->surfaceList[0].planeParams.pitch[0];
    const size_t spitch_uv = surf->surfaceList[0].planeParams.pitch[1];
    auto * dst = static_cast<uint8_t *>(cuda_dst);
    // Y plane: ew columns × h rows, src stride may have HW padding
    cudaMemcpy2D(dst,             static_cast<size_t>(ew), y_src,  spitch_y,
                 static_cast<size_t>(ew), static_cast<size_t>(h),
                 cudaMemcpyHostToDevice);
    // UV plane: ew columns × (h/2) rows (interleaved U/V, each row is ew bytes)
    cudaMemcpy2D(dst + ew * h,    static_cast<size_t>(ew), uv_src, spitch_uv,
                 static_cast<size_t>(ew), static_cast<size_t>(h / 2),
                 cudaMemcpyHostToDevice);
  };
  copy_nv12_to_cuda(nvbuf_left_,  cuda_nv12_left_,  ew, combined_height_);
  copy_nv12_to_cuda(nvbuf_right_, cuda_nv12_right_, ew, combined_height_);

  RCLCPP_INFO_ONCE(get_logger(),
    "NITROS NV12 publish via cudaMemcpy2D bridge:"
    " cuda_nv12_left=%p  cuda_nv12_right=%p  eye=%dx%d",
    cuda_nv12_left_, cuda_nv12_right_, ew, combined_height_);

  // ── Always publish camera_info ────────────────────────────────────────────
  auto pub_cam_info = [&](
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr & info_pub,
    const sensor_msgs::msg::CameraInfo & tpl,
    const std::string & frame_id)
  {
    auto info          = tpl;
    info.header.stamp    = stamp;
    info.header.frame_id = frame_id;
    info_pub->publish(info);
  };
  pub_cam_info(pub_left_info_,  cam_info_left_,  frame_id_left_);
  pub_cam_info(pub_right_info_, cam_info_right_, frame_id_right_);

  // ── Raw sensor_msgs/Image (rviz2) ─────────────────────────────────────────
  // VIC hw path (preferred): nvbuf_raw_{left,right}_ already contain BGRA data
  // from the async transform above. mappedAddr.addr[0] is directly usable as a
  // cv::Mat — no heap allocation, no memcpy, no cvtColor loop.
  //
  // CPU fallback (if BGRA surfaces weren't allocated): de-stride the NV12
  // SURFACE_ARRAY and run cv::cvtColor — same as the original path.
  auto publish_raw_side = [&](
    NvBufSurface * bgra_surf,        // null → CPU fallback
    NvBufSurface * nv12_surf,        // used for CPU fallback only
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr & pub,
    const std::string & frame_id,
    int out_w, int out_h)
  {
    cv::Mat bgr;

    if (bgra_surf) {
      // ── VIC path: wrap BGRA directly, convert BGRA→BGR (single NEON pass) ──
      auto * bgra_ptr = static_cast<uint8_t *>(
        bgra_surf->surfaceList[0].mappedAddr.addr[0]);
      const size_t pitch = bgra_surf->surfaceList[0].planeParams.pitch[0];
      cv::Mat bgra_mat(combined_height_, ew, CV_8UC4, bgra_ptr, pitch);
      cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);  // drops X channel; NEON-vectorised
    } else {
      // ── CPU fallback: de-stride NV12 + cvtColor ────────────────────────────
      auto * y_src  = static_cast<const uint8_t *>(
        nv12_surf->surfaceList[0].mappedAddr.addr[0]);
      auto * uv_src = static_cast<const uint8_t *>(
        nv12_surf->surfaceList[0].mappedAddr.addr[1]);
      const size_t pitch_y  = nv12_surf->surfaceList[0].planeParams.pitch[0];
      const size_t pitch_uv = nv12_surf->surfaceList[0].planeParams.pitch[1];
      std::vector<uint8_t> nv12_buf(static_cast<size_t>(ew * combined_height_ * 3 / 2));
      for (int r = 0; r < combined_height_; ++r) {
        std::memcpy(nv12_buf.data() + r * ew, y_src + r * pitch_y,
                    static_cast<size_t>(ew));
      }
      for (int r = 0; r < combined_height_ / 2; ++r) {
        std::memcpy(nv12_buf.data() + ew * combined_height_ + r * ew,
                    uv_src + r * pitch_uv, static_cast<size_t>(ew));
      }
      cv::Mat nv12_mat(combined_height_ * 3 / 2, ew, CV_8UC1, nv12_buf.data());
      cv::cvtColor(nv12_mat, bgr, cv::COLOR_YUV2BGR_NV12);
    }

    if (out_w != bgr.cols || out_h != bgr.rows) {
      cv::resize(bgr, bgr, cv::Size(out_w, out_h));
    }
    auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header{}, "bgr8", bgr).toImageMsg();
    img_msg->header.stamp    = stamp;
    img_msg->header.frame_id = frame_id;
    pub->publish(std::make_unique<sensor_msgs::msg::Image>(std::move(*img_msg)));
  };

  if (pub_raw_left_)
    publish_raw_side(have_raw_surfs ? nvbuf_raw_left_  : nullptr, nvbuf_left_,
                     pub_left_raw_,  frame_id_left_,  eff_out_w(true),  eff_out_h(true));
  if (pub_raw_right_)
    publish_raw_side(have_raw_surfs ? nvbuf_raw_right_ : nullptr, nvbuf_right_,
                     pub_right_raw_, frame_id_right_, eff_out_w(false), eff_out_h(false));

  // ── NITROS publish (GPU-resident zero-copy) ───────────────────────────────
  // cuda_nv12_left_/right_ are real cudaMalloc device pointers for downstream
  // NITROS nodes (e.g. isaac_ros_image_proc, DNN inference pipelines).
  // Encoding (and topic suffix) are set by cam_{left,right}.nitros_format param.

  auto make_header = [&](const std::string & frame_id) {
    std_msgs::msg::Header h;
    h.stamp    = stamp;
    h.frame_id = frame_id;
    return h;
  };

  // NitrosImage carries the requested out_w/out_h dimensions in the header;
  // downstream NITROS resize node handles further scaling on GPU if needed.
  auto publish_nitros = [&](
    void * cuda_ptr,
    rclcpp::Publisher<NitrosImage>::SharedPtr & pub,
    const std::string & frame_id,
    const std::string & encoding,
    int out_w, int out_h)
  {
    NitrosImage nitros_img = NitrosImageBuilder()
      .WithHeader(make_header(frame_id))
      .WithEncoding(encoding)
      .WithDimensions(static_cast<uint32_t>(out_h),
                      static_cast<uint32_t>(out_w))
      .WithGpuData(cuda_ptr)
      .Build();
    pub->publish(std::move(nitros_img));
  };

  publish_nitros(cuda_nv12_left_,  pub_left_nitros_,
                 frame_id_left_,  nitros_fmt_left_,  eff_out_w(true),  eff_out_h(true));
  publish_nitros(cuda_nv12_right_, pub_right_nitros_,
                 frame_id_right_, nitros_fmt_right_, eff_out_w(false), eff_out_h(false));

  return true;
}

#endif  // HAVE_NVBUF

// ─────────────────────────────────────────────────────────────────────────────
// capture_loop()  — runs in a dedicated thread
// ─────────────────────────────────────────────────────────────────────────────
void ArducamDualCamNode::capture_loop()
{
  RCLCPP_INFO(get_logger(), "Capture thread started");

  while (running_) {
    GstSample * sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink_));
    if (!sample) {
      if (running_) {
        RCLCPP_WARN(get_logger(), "pull_sample returned null (EOS or error)");
      }
      break;
    }
    process_sample(sample);
    gst_sample_unref(sample);
  }

  RCLCPP_INFO(get_logger(), "Capture thread exited");
}

// ─────────────────────────────────────────────────────────────────────────────
// process_sample()
// ─────────────────────────────────────────────────────────────────────────────
void ArducamDualCamNode::process_sample(GstSample * sample)
{
  GstBuffer * buf = gst_sample_get_buffer(sample);
  if (!buf) return;

  // ── Hardware timestamp ────────────────────────────────────────────────────
  // v4l2src copies the V4L2 kernel buffer timestamp (CLOCK_MONOTONIC) into
  // the GStreamer buffer PTS.  Both left and right messages from the same
  // combined frame get the same PTS — stereo frames are co-stamped precisely.
  GstClockTime pts = GST_BUFFER_PTS(buf);
  rclcpp::Time stamp;
  if (GST_CLOCK_TIME_IS_VALID(pts)) {
    stamp = rclcpp::Time(static_cast<int64_t>(pts) + gst_clock_offset_, RCL_SYSTEM_TIME);
  } else {
    RCLCPP_WARN_ONCE(get_logger(), "No valid PTS — falling back to rclcpp::now()");
    stamp = now();
  }

  // ── VIC NvBufSurfTransform path ──────────────────────────────────────────
  // When the NVMM pipeline and NvBufSurface are both available, hand off to
  // the VIC path which does zero-copy crop + colour conversion in hardware.
  // process_sample_nvbuf() performs its own gst_buffer_map/unmap internally.
#ifdef HAVE_NVBUF
  if (use_nvbuf_) {
    if (process_sample_nvbuf(buf, stamp)) {
      return;  // VIC path completed successfully — done for this frame
    }
    // process_sample_nvbuf() returned false on permanent VIC failure
    // (use_nvbuf_ is now false).  The NVMM pipeline is still running and
    // gst_buffer_map() will only return 64 bytes of NvBufSurface* — not pixel
    // data.  Rebuild to the system-memory BGRx pipeline before continuing.
    RCLCPP_WARN(get_logger(),
      "VIC NvBufSurf path failed — rebuilding to system-memory BGRx pipeline");
    if (use_nvmm_) {
      gst_element_set_state(pipeline_, GST_STATE_NULL);
      gst_object_unref(appsink_);  appsink_  = nullptr;
      gst_object_unref(pipeline_); pipeline_ = nullptr;
      use_nvmm_ = false;
      if (!try_build_pipeline(/*nvmm=*/false, "BGRx")) {
        RCLCPP_FATAL(get_logger(), "CPU pipeline rebuild failed — shutting down");
        running_ = false;
      }
    }
    return;  // skip this sample; CPU path will handle the next frame
  }
#endif

  // ── CPU fallback path (system-memory BGRx pipeline) ──────────────────────
  // This path is used when:
  //   • HAVE_NVBUF is not compiled in, OR
  //   • use_nvbuf_ is false (NvBufSurface init failed), OR
  //   • process_sample_nvbuf() returned false (permanent VIC error)
  //
  // The system-memory pipeline always outputs BGRx (4 B/px).  The combined
  // frame is split into left/right via a zero-copy cv::Mat ROI, then
  // BGRx→BGR8 is performed per-eye.  Compiler flags -O3 + armv8.2-a+simd
  // auto-vectorise these loops with NEON.

  // ── Stride from GstVideoInfo ──────────────────────────────────────────────
  // nvvidconv aligns DMA buffer rows (often to 64–128 bytes).
  GstCaps * caps = gst_sample_get_caps(sample);
  GstVideoInfo vinfo;
  gst_video_info_init(&vinfo);
  size_t stride = static_cast<size_t>(combined_width_) * 4;  // packed fallback
  if (caps && gst_video_info_from_caps(&vinfo, caps)) {
    stride = static_cast<size_t>(GST_VIDEO_INFO_PLANE_STRIDE(&vinfo, 0));
    RCLCPP_INFO_ONCE(get_logger(),
      "CPU path — video info: %dx%d  stride=%zu  (nominal row=%d bytes)",
      GST_VIDEO_INFO_WIDTH(&vinfo), GST_VIDEO_INFO_HEIGHT(&vinfo),
      stride, combined_width_ * 4);
  } else {
    RCLCPP_WARN_ONCE(get_logger(),
      "Could not parse GstVideoInfo; using packed stride %zu", stride);
  }

  GstMapInfo map{};
  if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
    RCLCPP_WARN(get_logger(), "Failed to map GStreamer buffer");
    return;
  }

  const size_t min_expected =
    stride * static_cast<size_t>(combined_height_ - 1)
    + static_cast<size_t>(combined_width_) * 4;
  if (map.size < min_expected) {
    RCLCPP_WARN_ONCE(get_logger(),
      "Buffer size mismatch: got %zu expected >= %zu (stride=%zu)",
      map.size, min_expected, stride);
    gst_buffer_unmap(buf, &map);
    return;
  }

  // Wrap the VIC output as a combined BGRx cv::Mat — no pixel copy
  const int ew = eye_width();
  cv::Mat combined(combined_height_, combined_width_, CV_8UC4, map.data, stride);

  // Split into left/right via ROI — still no pixel copy
  cv::Mat left_bgra  = combined(cv::Rect(0,  0, ew, combined_height_));
  cv::Mat right_bgra = combined(cv::Rect(ew, 0, ew, combined_height_));

  // BGRx → BGR8 (drop the unused X channel, single pass each).
  // NEON-vectorised by -O3 -march=armv8.2-a+simd; outputs contiguous Mats.
  cv::Mat left_bgr, right_bgr;
  cv::cvtColor(left_bgra,  left_bgr,  cv::COLOR_BGRA2BGR);
  cv::cvtColor(right_bgra, right_bgr, cv::COLOR_BGRA2BGR);

  gst_buffer_unmap(buf, &map);

  // ── Publish camera_info (always) + image_raw (per-flag) ──────────────────
  // camera_info is always emitted regardless of publish_image_raw.
  {
    auto publish_info = [&](
      rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr & info_pub,
      const sensor_msgs::msg::CameraInfo & tpl,
      const std::string & frame_id)
    {
      auto info          = tpl;
      info.header.stamp    = stamp;
      info.header.frame_id = frame_id;
      info_pub->publish(info);
    };
    publish_info(pub_left_info_,  cam_info_left_,  frame_id_left_);
    publish_info(pub_right_info_, cam_info_right_, frame_id_right_);
  }

  // image_raw published per-side only when publish_image_raw is enabled
  auto publish_cpu_raw = [&](
    cv::Mat & bgr,
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr & img_pub,
    const std::string & frame_id,
    int out_w, int out_h)
  {
    cv::Mat final_bgr;
    if (out_w == bgr.cols && out_h == bgr.rows) {
      final_bgr = bgr;
    } else {
      cv::resize(bgr, final_bgr, cv::Size(out_w, out_h));
    }
    auto sp = cv_bridge::CvImage(std_msgs::msg::Header{}, "bgr8", final_bgr).toImageMsg();
    sp->header.stamp    = stamp;
    sp->header.frame_id = frame_id;
    img_pub->publish(std::make_unique<sensor_msgs::msg::Image>(std::move(*sp)));
  };

  if (pub_raw_left_)  publish_cpu_raw(left_bgr,  pub_left_raw_,  frame_id_left_,
                        eff_out_w(true),  eff_out_h(true));
  if (pub_raw_right_) publish_cpu_raw(right_bgr, pub_right_raw_, frame_id_right_,
                        eff_out_w(false), eff_out_h(false));
}

}  // namespace arducam_dual_camera

RCLCPP_COMPONENTS_REGISTER_NODE(arducam_dual_camera::ArducamDualCamNode)
