/**
 * MIT License ------------------------------------------------------------------------
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

  ------------------------------------------------------------------------------------

  * @file arducam_b0573_node.cpp
  * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
  * @date: 27.02.2026
  * @brief Single ROS 2 composable node for the Arducam B0573 (GMSL2-to-CSI2) dual camera.
*/


#include "isaac_ros_arducam_b0573/arducam_b0573_node.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// NITROS: NitrosImage is a TypeAdapter over sensor_msgs::msg::Image.
// Publishing sensor_msgs::msg::Image on an rclcpp::Publisher<NitrosImage>
// triggers convert_to_custom() in the TypeAdapter, which creates a
// lifecycle-managed GXF VideoBuffer (freed after all subscribers consume it).
// This avoids the 16-entity pool exhaustion from NitrosImageBuilder::Build().
#ifdef HAVE_NVBUF
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
using nvidia::isaac_ros::nitros::NitrosImage;

// CUDA remap kernel (defined in rectification_kernels.cu, compiled as a CUDA TU).
// Both d_src and d_dst must be packed BGRA device buffers (pitch = w*4).
// Function is asynchronous; caller synchronises the stream.
extern "C" void cuda_remap_bgra(
  const uint8_t* d_src, uint8_t* d_dst,
  const float* d_map_x, const float* d_map_y,
  int w, int h, cudaStream_t stream);
#endif

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <rclcpp_components/register_node_macro.hpp>

namespace nvidia
{
namespace isaac_ros
{
namespace arducam
{

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
ArducamB0573Node::ArducamB0573Node(const rclcpp::NodeOptions & options)
: Node("arducam_b0573_node", options)
{
  // ── Global parameters ─────────────────────────────────────────────────────
  declare_parameter("device",        "/dev/video0");
  declare_parameter("width",         2560);
  declare_parameter("height",        720);
  declare_parameter("fps",           30);
  declare_parameter("pixel_format",  "UYVY");

  // ── Per-camera parameters (left_camera / right_camera) ─────────────────────
  for (const std::string side : {"left_camera", "right_camera"}) {
    const bool is_left = (side == "left_camera");
    declare_parameter(side + ".frame_id",
      is_left ? "left_camera" : "right_camera");
    // GPU lens-distortion rectification (CUDA bilinear remap after VIC crop)
    declare_parameter(side + ".rectilinear", false);

    // ── topics.visual_stream ────────────────────────────────────────────────
    declare_parameter(side + ".topics.topic_name_prefix",
      is_left ? "/arducam/left" : "/arducam/right");
    declare_parameter(side + ".topics.visual_stream.enable",       true);
    declare_parameter(side + ".topics.visual_stream.transport",    std::string("compressed"));
    declare_parameter(side + ".topics.visual_stream.encoding",     std::string("bgr8"));
    declare_parameter(side + ".topics.visual_stream.jpeg_quality", 80);
    declare_parameter(side + ".topics.visual_stream.qos.reliability", std::string("reliable"));
    declare_parameter(side + ".topics.visual_stream.qos.durability",  std::string("volatile"));
    declare_parameter(side + ".topics.visual_stream.resolution.width",  -1);
    declare_parameter(side + ".topics.visual_stream.resolution.height", -1);

    // ── topics.nitros_image ─────────────────────────────────────────────────
    declare_parameter(side + ".topics.nitros_image.enable",    true);
    declare_parameter(side + ".topics.nitros_image.qos.reliability", std::string("reliable"));
    declare_parameter(side + ".topics.nitros_image.qos.durability",  std::string("volatile"));
    // FIX 1.1: default changed from "nv12" to "rgb8" — nv12 via TypeAdapter is broken
    // (TypeAdapter cudaMemcpy2D only copies the Y plane; UV stays uninitialized → green screen)
    declare_parameter(side + ".topics.nitros_image.format",    std::string("rgb8"));
    declare_parameter(side + ".topics.nitros_image.fps",       30);
    declare_parameter(side + ".topics.nitros_image.resolution.width",  -1);
    declare_parameter(side + ".topics.nitros_image.resolution.height", -1);

    // ── topics.camera_info ──────────────────────────────────────────────────
    declare_parameter(side + ".topics.camera_info.qos.reliability", std::string("reliable"));
    declare_parameter(side + ".topics.camera_info.qos.durability",  std::string("volatile"));
    declare_parameter(side + ".topics.camera_info.fps",        30);

    // ── extrinsics ───────────────────────────────────────────────────────────
    declare_parameter(side + ".extrinsics.relative_to",             "base_link");
    declare_parameter(side + ".extrinsics.rotation",    std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter(side + ".extrinsics.translation", std::vector<double>{0.0, 0.0, 0.0});

    // ── intrinsics ───────────────────────────────────────────────────────────
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

  // ── Read per-camera visual_stream config ──────────────────────────────────
  frame_id_left_  = get_parameter("left_camera.frame_id").as_string();
  frame_id_right_ = get_parameter("right_camera.frame_id").as_string();
  rectify_left_   = get_parameter("left_camera.rectilinear").as_bool();
  rectify_right_  = get_parameter("right_camera.rectilinear").as_bool();

  vis_enable_left_       = get_parameter("left_camera.topics.visual_stream.enable").as_bool();
  vis_enable_right_      = get_parameter("right_camera.topics.visual_stream.enable").as_bool();
  vis_transport_left_    = get_parameter("left_camera.topics.visual_stream.transport").as_string();
  vis_transport_right_   = get_parameter("right_camera.topics.visual_stream.transport").as_string();
  vis_encoding_left_     = get_parameter("left_camera.topics.visual_stream.encoding").as_string();
  vis_encoding_right_    = get_parameter("right_camera.topics.visual_stream.encoding").as_string();
  vis_jpeg_quality_left_  = get_parameter("left_camera.topics.visual_stream.jpeg_quality").as_int();
  vis_jpeg_quality_right_ = get_parameter("right_camera.topics.visual_stream.jpeg_quality").as_int();

  // Derive topic names from prefix + transport
  {
    const std::string pfx_l = get_parameter("left_camera.topics.topic_name_prefix").as_string();
    const std::string pfx_r = get_parameter("right_camera.topics.topic_name_prefix").as_string();
    topic_vis_left_   = (vis_transport_left_  == "compressed") ? pfx_l + "/image/compressed"
                                                                : pfx_l + "/image_raw";
    topic_vis_right_  = (vis_transport_right_ == "compressed") ? pfx_r + "/image/compressed"
                                                               : pfx_r + "/image_raw";
    topic_left_info_  = (vis_transport_left_  == "compressed") ? pfx_l + "/image/camera_info"
                                                                : pfx_l + "/camera_info";
    // FIX 1.2: was incorrectly using vis_transport_left_ — right side topic depended on left's transport
    topic_right_info_ = (vis_transport_right_ == "compressed") ? pfx_r + "/image/camera_info"
                                                                : pfx_r + "/camera_info";
  }

  // visual_stream output resolution
  out_w_vis_left_   = get_parameter("left_camera.topics.visual_stream.resolution.width").as_int();
  out_h_vis_left_   = get_parameter("left_camera.topics.visual_stream.resolution.height").as_int();
  out_w_vis_right_  = get_parameter("right_camera.topics.visual_stream.resolution.width").as_int();
  out_h_vis_right_  = get_parameter("right_camera.topics.visual_stream.resolution.height").as_int();

  // nitros_image output resolution
  out_w_nitros_left_  = get_parameter("left_camera.topics.nitros_image.resolution.width").as_int();
  out_h_nitros_left_  = get_parameter("left_camera.topics.nitros_image.resolution.height").as_int();
  out_w_nitros_right_ = get_parameter("right_camera.topics.nitros_image.resolution.width").as_int();
  out_h_nitros_right_ = get_parameter("right_camera.topics.nitros_image.resolution.height").as_int();

  qos_vis_rel_left_     = get_parameter("left_camera.topics.visual_stream.qos.reliability").as_string();
  qos_vis_dur_left_     = get_parameter("left_camera.topics.visual_stream.qos.durability").as_string();
  qos_vis_rel_right_    = get_parameter("right_camera.topics.visual_stream.qos.reliability").as_string();
  qos_vis_dur_right_    = get_parameter("right_camera.topics.visual_stream.qos.durability").as_string();
  qos_info_rel_left_    = get_parameter("left_camera.topics.camera_info.qos.reliability").as_string();
  qos_info_dur_left_    = get_parameter("left_camera.topics.camera_info.qos.durability").as_string();
  qos_info_rel_right_   = get_parameter("right_camera.topics.camera_info.qos.reliability").as_string();
  qos_info_dur_right_   = get_parameter("right_camera.topics.camera_info.qos.durability").as_string();
  qos_nitros_rel_left_  = get_parameter("left_camera.topics.nitros_image.qos.reliability").as_string();
  qos_nitros_dur_left_  = get_parameter("left_camera.topics.nitros_image.qos.durability").as_string();
  qos_nitros_rel_right_ = get_parameter("right_camera.topics.nitros_image.qos.reliability").as_string();
  qos_nitros_dur_right_ = get_parameter("right_camera.topics.nitros_image.qos.durability").as_string();

  pub_nitros_left_      = get_parameter("left_camera.topics.nitros_image.enable").as_bool();
  pub_nitros_right_     = get_parameter("right_camera.topics.nitros_image.enable").as_bool();
  nitros_fmt_left_      = get_parameter("left_camera.topics.nitros_image.format").as_string();
  nitros_fmt_right_     = get_parameter("right_camera.topics.nitros_image.format").as_string();

  // FIX 1.1: nv12 via TypeAdapter only copies the Y plane; UV plane in the GXF
  // VideoBuffer is never populated → downstream receives random chroma.  Refuse
  // early with a clear message rather than producing a silent green-screen image.
  {
    const bool bad_l = pub_nitros_left_  && nitros_fmt_left_  == "nv12";
    const bool bad_r = pub_nitros_right_ && nitros_fmt_right_ == "nv12";
    if (bad_l || bad_r) {
      RCLCPP_FATAL(get_logger(),
        "nitros_image.format='nv12' is NOT supported via the TypeAdapter path: "
        "convert_to_custom() calls a single cudaMemcpy2D(height=img.height) which "
        "only copies the Y plane; the UV plane in the GXF VideoBuffer is never "
        "written and downstream nodes see random/zero chroma (green screen). "
        "Set format to 'rgb8' or 'bgr8' instead. "
        "(left=%s, right=%s)",
        nitros_fmt_left_.c_str(), nitros_fmt_right_.c_str());
      throw std::runtime_error("Unsupported nitros_image.format='nv12'");
    }
  }

  RCLCPP_INFO(get_logger(),
    "Arducam B0573 | device=%s  combined=%dx%d  eye=%dx%d  fps=%d  fmt=%s",
    device_.c_str(), combined_width_, combined_height_,
    eye_width(), combined_height_, fps_, pixel_format_.c_str());
  RCLCPP_INFO(get_logger(),
    "Left  | vis=%s[%s/%s]  nitros=%s(%s)  frame=%s  out_vis=%dx%d  out_nitros=%dx%d",
    vis_enable_left_  ? topic_vis_left_.c_str()    : "(disabled)",
    vis_transport_left_.c_str(), vis_encoding_left_.c_str(),
    pub_nitros_left_  ? topic_left_nitros_.c_str() : "(disabled)",
    nitros_fmt_left_.c_str(),
    frame_id_left_.c_str(),
    eff_out_w(true),        eff_out_h(true),
    eff_out_w_nitros(true), eff_out_h_nitros(true));
  RCLCPP_INFO(get_logger(),
    "        vis_qos=[%s/%s]  info_qos=[%s/%s]  nitros_qos=[%s/%s]",
    qos_vis_rel_left_.c_str(),    qos_vis_dur_left_.c_str(),
    qos_info_rel_left_.c_str(),   qos_info_dur_left_.c_str(),
    qos_nitros_rel_left_.c_str(), qos_nitros_dur_left_.c_str());
  RCLCPP_INFO(get_logger(),
    "Right | vis=%s[%s/%s]  nitros=%s(%s)  frame=%s  out_vis=%dx%d  out_nitros=%dx%d",
    vis_enable_right_ ? topic_vis_right_.c_str()    : "(disabled)",
    vis_transport_right_.c_str(), vis_encoding_right_.c_str(),
    pub_nitros_right_ ? topic_right_nitros_.c_str() : "(disabled)",
    nitros_fmt_right_.c_str(),
    frame_id_right_.c_str(),
    eff_out_w(false),        eff_out_h(false),
    eff_out_w_nitros(false), eff_out_h_nitros(false));
  RCLCPP_INFO(get_logger(),
    "        vis_qos=[%s/%s]  info_qos=[%s/%s]  nitros_qos=[%s/%s]",
    qos_vis_rel_right_.c_str(),    qos_vis_dur_right_.c_str(),
    qos_info_rel_right_.c_str(),   qos_info_dur_right_.c_str(),
    qos_nitros_rel_right_.c_str(), qos_nitros_dur_right_.c_str());

  // ── Publishers (per-camera QoS) ─────────────────────────────────────────
  auto make_qos = [](const std::string & rel, const std::string & dur) {
    rclcpp::QoS q(rclcpp::KeepLast(10));
    (rel == "reliable") ? q.reliable()    : q.best_effort();
    (dur == "transient_local") ? q.transient_local() : q.durability_volatile();
    return q;
  };

  // ── visual_stream publishers — created only when enable=true ──────────────
  if (vis_enable_left_) {
    if (vis_transport_left_ == "compressed") {
      pub_vis_comp_left_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        topic_vis_left_, make_qos(qos_vis_rel_left_, qos_vis_dur_left_));
      RCLCPP_INFO(get_logger(), "visual_stream LEFT  -> %s  [compressed/%s  q=%d]",
        topic_vis_left_.c_str(), vis_encoding_left_.c_str(), vis_jpeg_quality_left_);
    } else {
      pub_vis_raw_left_ = create_publisher<sensor_msgs::msg::Image>(
        topic_vis_left_, make_qos(qos_vis_rel_left_, qos_vis_dur_left_));
      RCLCPP_INFO(get_logger(), "visual_stream LEFT  -> %s  [raw/%s]",
        topic_vis_left_.c_str(), vis_encoding_left_.c_str());
    }
  } else {
    RCLCPP_INFO(get_logger(), "visual_stream LEFT   disabled (topics.visual_stream.enable=false)");
  }
  if (vis_enable_right_) {
    if (vis_transport_right_ == "compressed") {
      pub_vis_comp_right_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        topic_vis_right_, make_qos(qos_vis_rel_right_, qos_vis_dur_right_));
      RCLCPP_INFO(get_logger(), "visual_stream RIGHT -> %s  [compressed/%s  q=%d]",
        topic_vis_right_.c_str(), vis_encoding_right_.c_str(), vis_jpeg_quality_right_);
    } else {
      pub_vis_raw_right_ = create_publisher<sensor_msgs::msg::Image>(
        topic_vis_right_, make_qos(qos_vis_rel_right_, qos_vis_dur_right_));
      RCLCPP_INFO(get_logger(), "visual_stream RIGHT -> %s  [raw/%s]",
        topic_vis_right_.c_str(), vis_encoding_right_.c_str());
    }
  } else {
    RCLCPP_INFO(get_logger(), "visual_stream RIGHT  disabled (topics.visual_stream.enable=false)");
  }
  // camera_info is always published
  pub_left_info_  = create_publisher<sensor_msgs::msg::CameraInfo>(
    topic_left_info_,  make_qos(qos_info_rel_left_,  qos_info_dur_left_));
  pub_right_info_ = create_publisher<sensor_msgs::msg::CameraInfo>(
    topic_right_info_, make_qos(qos_info_rel_right_, qos_info_dur_right_));

  // ── NITROS publishers (GPU-resident zero-copy, HAVE_NVBUF only) ────────────────
  // Topic name is fully specified by {left,right}_camera.topics.nitros_image.topic_name.
  // Supported NitrosImageBuilder encodings: "nv12", "rgb8", "bgr8"
#ifdef HAVE_NVBUF
  topic_left_nitros_  = get_parameter("left_camera.topics.topic_name_prefix").as_string()
                        + "/nitros_image_" + nitros_fmt_left_;
  topic_right_nitros_ = get_parameter("right_camera.topics.topic_name_prefix").as_string()
                        + "/nitros_image_" + nitros_fmt_right_;
  if (pub_nitros_left_) {
    pub_left_nitros_  = create_publisher<NitrosImage>(
      topic_left_nitros_,  make_qos(qos_nitros_rel_left_,  qos_nitros_dur_left_));
    RCLCPP_INFO(get_logger(), "NITROS LEFT  -> %s  (format=%s)",
      topic_left_nitros_.c_str(), nitros_fmt_left_.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "NITROS LEFT   disabled (topics.nitros_image.enable=false)");
  }
  if (pub_nitros_right_) {
    pub_right_nitros_ = create_publisher<NitrosImage>(
      topic_right_nitros_, make_qos(qos_nitros_rel_right_, qos_nitros_dur_right_));
    RCLCPP_INFO(get_logger(), "NITROS RIGHT -> %s  (format=%s)",
      topic_right_nitros_.c_str(), nitros_fmt_right_.c_str());
  } else {
    RCLCPP_INFO(get_logger(), "NITROS RIGHT  disabled (topics.nitros_image.enable=false)");
  }
#endif

  // ── Build CameraInfo from inline intrinsic parameters ────────────────────
  cam_info_left_  = build_cam_info("left_camera",  eye_width(), combined_height_);
  cam_info_right_ = build_cam_info("right_camera", eye_width(), combined_height_);

  // ── Broadcast static TF transforms (extrinsics) ──────────────────────────
  tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
  broadcast_static_tf("left_camera",  frame_id_left_);
  broadcast_static_tf("right_camera", frame_id_right_);

  // ── GStreamer ────────────────────────────────────────────────────────────
  int   argc = 0;
  char ** argv = nullptr;
  gst_init(&argc, &argv);

#ifdef HAVE_NVBUF
  // No explicit GXF context init needed: the NitrosImage TypeAdapter registers
  // its own context when the first subscriber negotiates format.
#endif

  build_pipeline();   // also calls init_nvbuf_surfaces() when HAVE_NVBUF

  running_        = true;
  capture_thread_ = std::thread(&ArducamB0573Node::capture_loop, this);
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
ArducamB0573Node::~ArducamB0573Node()
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
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// build_cam_info()
//   Reads the inline intrinsic parameters for the given camera side
//   ("left_camera" or "right_camera") and constructs a sensor_msgs::CameraInfo.
//   The header (stamp / frame_id) is intentionally left empty here; it is filled
//   in at publish time so that each message carries the correct per-frame stamp.
// ─────────────────────────────────────────────────────────────────────────────
sensor_msgs::msg::CameraInfo ArducamB0573Node::build_cam_info(
  const std::string & side, int w, int h)
{
  const std::string p = side + ".intrinsics";
  sensor_msgs::msg::CameraInfo ci;
  ci.width  = static_cast<uint32_t>(w);
  ci.height = static_cast<uint32_t>(h);

  ci.distortion_model =
    get_parameter(p + ".distortion_model").as_string();

  auto d = get_parameter(p + ".distortion_coefficients").as_double_array();
  ci.d.assign(d.begin(), d.end());

  const double fx = get_parameter(p + ".fx").as_double();
  const double fy = get_parameter(p + ".fy").as_double();
  const double cx = get_parameter(p + ".cx").as_double();
  const double cy = get_parameter(p + ".cy").as_double();

  // Camera matrix K (3×3, row-major)
  ci.k = {fx,  0.0, cx,
          0.0, fy,  cy,
          0.0, 0.0, 1.0};

  // Rectification matrix R (3×3, row-major)
  auto r_data = get_parameter(p + ".reflection_matrix.data").as_double_array();
  if (r_data.size() == 9) {
    std::copy(r_data.begin(), r_data.end(), ci.r.begin());
  } else {
    ci.r = {1, 0, 0,  0, 1, 0,  0, 0, 1};
  }

  // Projection matrix P (3×4, row-major)
  auto p_data = get_parameter(p + ".projection_matrix.data").as_double_array();
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
//   Reads {left,right}_camera.topics.extrinsics and publishes a latched static TF
//   transform from the parent frame (extrinsics.relative_to) to child_frame.
//   rotation is [roll, pitch, yaw] in degrees; translation is [x, y, z] in m.
// ─────────────────────────────────────────────────────────────────────────────
void ArducamB0573Node::broadcast_static_tf(
  const std::string & side, const std::string & child_frame)
{
  const std::string ep = side + ".extrinsics";
  const std::string parent =
    get_parameter(ep + ".relative_to").as_string();
  auto rpy = get_parameter(ep + ".rotation").as_double_array();
  auto xyz = get_parameter(ep + ".translation").as_double_array();

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
bool ArducamB0573Node::try_build_pipeline(bool nvmm, const std::string & out_fmt)
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
void ArducamB0573Node::build_pipeline()
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
void ArducamB0573Node::init_nvbuf_surfaces()
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
  // FIX 1.3: track each map individually so we can UnMap before Destroy on
  // partial failure — destroying a still-mapped surface leaks the kernel nvmap handle.
  const bool map_left_ok  = (NvBufSurfaceMap(nvbuf_left_,  0, -1, NVBUF_MAP_READ_WRITE) == 0);
  const bool map_right_ok = map_left_ok &&
                            (NvBufSurfaceMap(nvbuf_right_, 0, -1, NVBUF_MAP_READ_WRITE) == 0);
  if (!map_left_ok || !map_right_ok) {
    RCLCPP_WARN(get_logger(),
      "NvBufSurfaceMap (NV12) failed — VIC path disabled");
    if (map_left_ok) NvBufSurfaceUnMap(nvbuf_left_, 0, -1);  // unmap before destroy
    NvBufSurfaceDestroy(nvbuf_left_);  nvbuf_left_  = nullptr;
    NvBufSurfaceDestroy(nvbuf_right_); nvbuf_right_ = nullptr;
    if (nvbuf_raw_left_)  { NvBufSurfaceDestroy(nvbuf_raw_left_);  nvbuf_raw_left_  = nullptr; }
    if (nvbuf_raw_right_) { NvBufSurfaceDestroy(nvbuf_raw_right_); nvbuf_raw_right_ = nullptr; }
    return;
  }

  // Map BGRA surfaces (single plane, planeIdx=-1 == 0 for packed formats, but
  // -1 is safe and consistent).
  // FIX 1.3 (BGRA): same pattern — track each map to unmap before Destroy on failure.
  if (nvbuf_raw_left_ && nvbuf_raw_right_) {
    const bool raw_left_ok  = (NvBufSurfaceMap(nvbuf_raw_left_,  0, -1, NVBUF_MAP_READ_WRITE) == 0);
    const bool raw_right_ok = raw_left_ok &&
                              (NvBufSurfaceMap(nvbuf_raw_right_, 0, -1, NVBUF_MAP_READ_WRITE) == 0);
    if (!raw_left_ok || !raw_right_ok) {
      RCLCPP_WARN(get_logger(),
        "NvBufSurfaceMap (BGRA) failed — image_raw will use CPU cvtColor fallback");
      if (raw_left_ok) NvBufSurfaceUnMap(nvbuf_raw_left_, 0, -1);  // unmap before destroy
      NvBufSurfaceDestroy(nvbuf_raw_left_);  nvbuf_raw_left_  = nullptr;
      NvBufSurfaceDestroy(nvbuf_raw_right_); nvbuf_raw_right_ = nullptr;
    }
  }

  use_nvbuf_ = true;

  // FIX 2.2: Pre-size the staging image data buffers once so that make_nitros_image's
  // data.resize() on the per-frame hot path is always a no-op (vector already right size).
  // nv12 is rejected in the constructor, so only rgb8/bgr8 (3 bytes/px) and mono8 (1 byte/px).
  auto pre_size_nitros_img = [&](sensor_msgs::msg::Image & img, bool is_left) {
    const int w = eff_out_w_nitros(is_left);
    const int h = eff_out_h_nitros(is_left);
    const std::string & fmt = is_left ? nitros_fmt_left_ : nitros_fmt_right_;
    img.width  = static_cast<uint32_t>(w);
    img.height = static_cast<uint32_t>(h);
    const size_t bytes = (fmt == "mono8")
                         ? static_cast<size_t>(w) * static_cast<size_t>(h)
                         : static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    img.data.resize(bytes);  // allocates once; subsequent same-size resize() are no-ops
  };
  pre_size_nitros_img(nitros_img_left_,  true);
  pre_size_nitros_img(nitros_img_right_, false);

  // FIX 2.3: Pre-size the shared NV12 CPU de-stride staging buffer.
  // Used only when BGRA VIC surfaces are unavailable. Size = eye_w × h × 3/2.
  nv12_cpu_staging_.resize(
    static_cast<size_t>(eye_width()) * static_cast<size_t>(combined_height_) * 3 / 2);

  // FIX 3.2: Register a persistent VIC session so NvBufSurfTransformAsync actually
  // enqueues work on Orin.  Without SetSessionParams, the call silently returns
  // Success but submits nothing — the sync object never fires and the dst surface
  // stays zero-initialised (green screen).
  vic_session_.compute_mode = NvBufSurfTransformCompute_Default;  // driver picks VIC
  vic_session_.gpu_id       = 0;
  vic_session_.cuda_stream  = nullptr;  // VIC, not CUDA
  NvBufSurfTransformSetSessionParams(&vic_session_);

  RCLCPP_INFO(get_logger(),
    "NvBufSurface ready  eye=%dx%d"
    "  NV12(NITROS): CPU-mapped via TypeAdapter"
    "  BGRA(image_raw): %s",
    eye_width(), combined_height_,
    (nvbuf_raw_left_ && nvbuf_raw_right_) ? "VIC hw path" : "CPU cvtColor fallback");

  // ── GPU rectification maps ────────────────────────────────────────────────
  // Precompute float32 remap tables once via OpenCV (CPU, < 1 ms).
  // Works even if one or both sides don't request rectification — the function
  // skips allocation for sides with rectify_{left,right}_ == false.
  init_rectification_maps();
}

// ── cleanup_nvbuf_surfaces() ──────────────────────────────────────────────────
void ArducamB0573Node::cleanup_nvbuf_surfaces()
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

  // ── CUDA rectification resources ──────────────────────────────────────────
  if (rect_stream_) { cudaStreamDestroy(rect_stream_);  rect_stream_ = nullptr; }
  if (d_rect_map_x_left_)  { cudaFree(d_rect_map_x_left_);  d_rect_map_x_left_  = nullptr; }
  if (d_rect_map_y_left_)  { cudaFree(d_rect_map_y_left_);  d_rect_map_y_left_  = nullptr; }
  if (d_rect_map_x_right_) { cudaFree(d_rect_map_x_right_); d_rect_map_x_right_ = nullptr; }
  if (d_rect_map_y_right_) { cudaFree(d_rect_map_y_right_); d_rect_map_y_right_ = nullptr; }
  if (d_rect_src_left_)    { cudaFree(d_rect_src_left_);    d_rect_src_left_    = nullptr; }
  if (d_rect_src_right_)   { cudaFree(d_rect_src_right_);   d_rect_src_right_   = nullptr; }
  if (d_rect_tmp_left_)    { cudaFree(d_rect_tmp_left_);    d_rect_tmp_left_    = nullptr; }
  if (d_rect_tmp_right_)   { cudaFree(d_rect_tmp_right_);   d_rect_tmp_right_   = nullptr; }

  use_nvbuf_ = false;
}

// ── init_rectification_maps() ─────────────────────────────────────────────────
// Called once from init_nvbuf_surfaces() after all NvBufSurfaces are ready.
//
// For each side where rectify_{left,right}_ = true:
//   1. Extract K (3×3), D (Nx1), R (3×3), P (3×4) from cam_info_{left,right}_.
//   2. cv::initUndistortRectifyMap → float32 map_x, map_y (CPU, ~0.5 ms).
//   3. cudaMalloc + cudaMemcpy H→D for both maps.
//   4. cudaMalloc packed BGRA staging (d_rect_src) and output (d_rect_tmp) buffers.
//   5. Create a non-blocking CUDA stream for async remap.
//
// A single rect_stream_ is shared between both sides.
// The fisheye model (thin_prism_fisheye / equidistant) is detected via
// distortion_model and routes to cv::fisheye::initUndistortRectifyMap.
void ArducamB0573Node::init_rectification_maps()
{
  if (!rectify_left_ && !rectify_right_) {
    return;  // nothing to do — avoid creating a CUDA stream unnecessarily
  }

  const int ew = eye_width();
  const int h  = combined_height_;
  const cv::Size img_size(ew, h);
  const size_t map_bytes  = static_cast<size_t>(ew) * static_cast<size_t>(h) * sizeof(float);
  const size_t bgra_bytes = static_cast<size_t>(ew) * static_cast<size_t>(h) * 4;

  // Helper: build remap maps for one calibration, upload to device, allocate bufs.
  auto build_side = [&](
    const sensor_msgs::msg::CameraInfo & ci,
    float ** d_map_x, float ** d_map_y,
    uint8_t ** d_src,  uint8_t ** d_tmp,
    const std::string & side_name) -> bool
  {
    // ── Camera matrix K ─────────────────────────────────────────────────────
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = ci.k[0]; K.at<double>(0, 2) = ci.k[2];
    K.at<double>(1, 1) = ci.k[4]; K.at<double>(1, 2) = ci.k[5];

    // ── Distortion coefficients D ────────────────────────────────────────────
    cv::Mat D(static_cast<int>(ci.d.size()), 1, CV_64F);
    for (size_t i = 0; i < ci.d.size(); ++i) D.at<double>(static_cast<int>(i), 0) = ci.d[i];

    // ── Rectification matrix R ───────────────────────────────────────────────
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    if (ci.r.size() == 9) {
      for (int i = 0; i < 9; ++i)
        R.at<double>(i / 3, i % 3) = ci.r[i];
    }

    // ── New camera matrix P' (first 3×3 of P) ───────────────────────────────
    // P is 3×4: [fx' 0 cx' Tx; 0 fy' cy' Ty; 0 0 1 0]
    // cv::initUndistortRectifyMap needs the 3×3 part only.
    cv::Mat P3(3, 3, CV_64F, cv::Scalar(0));
    if (ci.p.size() == 12) {
      P3.at<double>(0, 0) = ci.p[0]; P3.at<double>(0, 2) = ci.p[2];
      P3.at<double>(1, 1) = ci.p[5]; P3.at<double>(1, 2) = ci.p[6];
      P3.at<double>(2, 2) = 1.0;
    } else {
      P3 = K.clone();  // fall back to using K if P not populated
    }

    // ── Compute remap tables on CPU (once, ~0.5 ms) ──────────────────────────
    cv::Mat map_x_cpu, map_y_cpu;
    const bool is_fisheye =
      ci.distortion_model == "equidistant" ||
      ci.distortion_model == "thin_prism_fisheye" ||
      ci.distortion_model == "kannala_brandt4";

    try {
      if (is_fisheye) {
        // cv::fisheye::initUndistortRectifyMap requires D to be a 1×4 or 4×1 vector.
        cv::Mat D4 = cv::Mat::zeros(4, 1, CV_64F);
        for (int i = 0; i < std::min(4, D.rows); ++i)
          D4.at<double>(i, 0) = D.at<double>(i, 0);
        cv::fisheye::initUndistortRectifyMap(K, D4, R, P3, img_size, CV_32FC1, map_x_cpu, map_y_cpu);
      } else {
        cv::initUndistortRectifyMap(K, D, R, P3, img_size, CV_32FC1, map_x_cpu, map_y_cpu);
      }
    } catch (const cv::Exception & e) {
      RCLCPP_ERROR(get_logger(),
        "init_rectification_maps [%s]: cv::initUndistortRectifyMap failed: %s"
        " — rectification disabled for this side", side_name.c_str(), e.what());
      return false;
    }

    // ── Upload maps to device ────────────────────────────────────────────────
    if (cudaMalloc(reinterpret_cast<void **>(d_map_x), map_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(d_map_y), map_bytes) != cudaSuccess)
    {
      RCLCPP_ERROR(get_logger(), "init_rectification_maps [%s]: cudaMalloc maps failed",
        side_name.c_str());
      if (*d_map_x) { cudaFree(*d_map_x); *d_map_x = nullptr; }
      return false;
    }
    cudaMemcpy(*d_map_x, map_x_cpu.data, map_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_map_y, map_y_cpu.data, map_bytes, cudaMemcpyHostToDevice);

    // ── Allocate packed BGRA device staging buffers ──────────────────────────
    // d_src: H2D packed copy of the VIC surface (de-strided) before remap.
    // d_tmp: remap output (packed BGRA), later copied back to the VIC surface.
    if (cudaMalloc(reinterpret_cast<void **>(d_src), bgra_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(d_tmp), bgra_bytes) != cudaSuccess)
    {
      RCLCPP_ERROR(get_logger(), "init_rectification_maps [%s]: cudaMalloc BGRA bufs failed",
        side_name.c_str());
      if (*d_src) { cudaFree(*d_src); *d_src = nullptr; }
      cudaFree(*d_map_x); *d_map_x = nullptr;
      cudaFree(*d_map_y); *d_map_y = nullptr;
      return false;
    }

    RCLCPP_INFO(get_logger(),
      "Rectification maps [%s]: %dx%d  model=%s%s  maps=%.1f KB on GPU",
      side_name.c_str(), ew, h, ci.distortion_model.c_str(),
      is_fisheye ? " (fisheye)" : "",
      static_cast<double>(map_bytes * 2) / 1024.0);
    return true;
  };

  // Create a single non-blocking CUDA stream shared by both sides
  cudaStreamCreateWithFlags(&rect_stream_, cudaStreamNonBlocking);

  if (rectify_left_) {
    const bool ok = build_side(cam_info_left_,
      &d_rect_map_x_left_, &d_rect_map_y_left_,
      &d_rect_src_left_,   &d_rect_tmp_left_,
      "left");
    if (!ok) {
      RCLCPP_WARN(get_logger(), "Left rectification disabled due to init failure");
      rectify_left_ = false;
    }
  }

  if (rectify_right_) {
    const bool ok = build_side(cam_info_right_,
      &d_rect_map_x_right_, &d_rect_map_y_right_,
      &d_rect_src_right_,   &d_rect_tmp_right_,
      "right");
    if (!ok) {
      RCLCPP_WARN(get_logger(), "Right rectification disabled due to init failure");
      rectify_right_ = false;
    }
  }
}

// ── apply_rect_bgra() ─────────────────────────────────────────────────────────
// Per-frame CUDA rectification of a BGRA NvBufSurface (in-place on mappedAddr).
//
// Steps:
//   1. cudaMemcpy2D: VIC surface (strided) → d_rect_src_  (packed, H2D)
//   2. cuda_remap_bgra kernel: d_rect_src_ → d_rect_tmp_  (packed → packed)
//   3. cudaStreamSynchronize
//   4. cudaMemcpy2D: d_rect_tmp_ (packed) → VIC surface  (strided, D2H)
//
// After step 4 the VIC surface's CPU-mapped addr contains the rectified pixels.
// The downstream publish_visual_side / make_nitros_image lambdas read from
// mappedAddr, so they automatically see the rectified data.
void ArducamB0573Node::apply_rect_bgra(NvBufSurface * surf, bool is_left)
{
  const int ew = eye_width();
  const int h  = combined_height_;
  const size_t packed_row  = static_cast<size_t>(ew) * 4;   // bytes/row in packed buffer
  const size_t surf_pitch  = surf->surfaceList[0].planeParams.pitch[0]; // VIC surface pitch

  uint8_t * const d_src = is_left ? d_rect_src_left_ : d_rect_src_right_;
  uint8_t * const d_tmp = is_left ? d_rect_tmp_left_ : d_rect_tmp_right_;
  const float * const d_mx = is_left ? d_rect_map_x_left_ : d_rect_map_x_right_;
  const float * const d_my = is_left ? d_rect_map_y_left_ : d_rect_map_y_right_;

  void * const cpu_addr = surf->surfaceList[0].mappedAddr.addr[0];

  // ── Step 1: H2D — packed copy from VIC surface (remove alignment padding) ──
  cudaMemcpy2D(
    d_src,                       // dst device (packed)
    packed_row,                  // dst pitch
    cpu_addr,                    // src host  (VIC strided)
    surf_pitch,                  // src pitch
    packed_row,                  // width in bytes (= ew * 4)
    static_cast<size_t>(h),
    cudaMemcpyHostToDevice);

  // ── Step 2: GPU remap (async) ─────────────────────────────────────────────
  cuda_remap_bgra(d_src, d_tmp, d_mx, d_my, ew, h, rect_stream_);

  // ── Step 3: Synchronise ───────────────────────────────────────────────────
  cudaStreamSynchronize(rect_stream_);

  // ── Step 4: D2H — write back to VIC surface (restore pitch alignment) ─────
  cudaMemcpy2D(
    cpu_addr,                    // dst host  (VIC strided)
    surf_pitch,                  // dst pitch
    d_tmp,                       // src device (packed)
    packed_row,                  // src pitch
    packed_row,                  // width in bytes
    static_cast<size_t>(h),
    cudaMemcpyDeviceToHost);
}

// ── process_sample_nvbuf() ────────────────────────────────────────────────────
// VIC + NITROS publishing path.
//
// Memory flow:
//   gst_buffer_map(NVMM) → NvBufSurface* src     [O(1), 64-byte struct]
//   NvBufSurfTransform (sync)  → VIC NV12 crop → nvbuf_left_ / nvbuf_right_
//     (SURFACE_ARRAY — the only dst type VIC accepts on Orin)
//   NvBufSurfTransformAsync    → VIC NV12→BGRA crop → nvbuf_raw_{left,right}_
//   Wait BGRA sync objects → gst_buffer_unmap
//   NvBufSurfaceSyncForCpu ×4  → ensure CPU view is coherent (all planes)
//   Build sensor_msgs::msg::Image from mappedAddr (std::memcpy, stride-strip)
//   pub_{left,right}_nitros_->publish(ros_img)  → TypeAdapter convert_to_custom()
//     creates a lifecycle-managed GXF VideoBuffer; freed after consumption.
//
// Returns false if any VIC operation fails; caller rebuilds to CPU pipeline.
bool ArducamB0573Node::process_sample_nvbuf(GstBuffer * buf, const rclcpp::Time & stamp)
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

  // ── FIX 3.1+3.2: All 4 VIC transforms submitted async concurrently ──────────
  // vic_session_ (set once in init_nvbuf_surfaces) enables NvBufSurfTransformAsync
  // on Orin.  Submitting NV12_L, NV12_R, BGRA_L, BGRA_R all before waiting lets
  // VIC pipeline them in its internal queue instead of blocking per-transform.
  // Each pair reads from the same src (independent writes to distinct dst surfaces).
  const bool want_raw     = vis_enable_left_ || vis_enable_right_
                          || rectify_left_  || rectify_right_;   // rectify needs BGRA surface
  const bool have_raw_surfs = nvbuf_raw_left_ && nvbuf_raw_right_;

  NvBufSurfTransformParams tp_raw_left{};
  NvBufSurfTransformParams tp_raw_right{};
  if (want_raw && have_raw_surfs) {
    tp_raw_left  = tp_left;
    tp_raw_right = tp_right;
    tp_raw_left.transform_flag  |= NVBUFSURF_TRANSFORM_FILTER;
    tp_raw_right.transform_flag |= NVBUFSURF_TRANSFORM_FILTER;
  }

  NvBufSurfTransformSyncObj_t sync_nv12_l = nullptr;
  NvBufSurfTransformSyncObj_t sync_nv12_r = nullptr;
  NvBufSurfTransformSyncObj_t sync_bgra_l = nullptr;
  NvBufSurfTransformSyncObj_t sync_bgra_r = nullptr;

  // Submit NV12 crops async (session ensures VIC actually receives them)
  auto err_l = NvBufSurfTransformAsync(src, nvbuf_left_,  &tp_left,  &sync_nv12_l);
  auto err_r = NvBufSurfTransformAsync(src, nvbuf_right_, &tp_right, &sync_nv12_r);
  if (err_l != NvBufSurfTransformError_Success ||
      err_r != NvBufSurfTransformError_Success)
  {
    RCLCPP_WARN_ONCE(get_logger(),
      "NvBufSurfTransformAsync (NV12) failed (err_l=%d err_r=%d) — "
      "disabling VIC path", static_cast<int>(err_l), static_cast<int>(err_r));
    if (sync_nv12_l) NvBufSurfTransformSyncObjDestroy(&sync_nv12_l);
    if (sync_nv12_r) NvBufSurfTransformSyncObjDestroy(&sync_nv12_r);
    gst_buffer_unmap(buf, &map);
    use_nvbuf_ = false;
    return false;
  }

  // Submit BGRA crops async — VIC queues behind NV12 jobs, overlaps with CPU work
  if (want_raw && have_raw_surfs) {
    auto err_bl = NvBufSurfTransformAsync(src, nvbuf_raw_left_,  &tp_raw_left,  &sync_bgra_l);
    auto err_br = NvBufSurfTransformAsync(src, nvbuf_raw_right_, &tp_raw_right, &sync_bgra_r);
    if (err_bl != NvBufSurfTransformError_Success ||
        err_br != NvBufSurfTransformError_Success) {
      RCLCPP_WARN_ONCE(get_logger(),
        "NvBufSurfTransformAsync (BGRA) failed (err_l=%d err_r=%d)"
        " — visual_stream will use CPU fallback",
        static_cast<int>(err_bl), static_cast<int>(err_br));
      if (sync_bgra_l) { NvBufSurfTransformSyncObjDestroy(&sync_bgra_l); sync_bgra_l = nullptr; }
      if (sync_bgra_r) { NvBufSurfTransformSyncObjDestroy(&sync_bgra_r); sync_bgra_r = nullptr; }
    }
  }

  // Wait for all submitted jobs then release src
  auto wait_and_destroy = [](NvBufSurfTransformSyncObj_t & s) {
    if (s) { NvBufSurfTransformSyncObjWait(s, -1); NvBufSurfTransformSyncObjDestroy(&s); }
  };
  wait_and_destroy(sync_nv12_l);
  wait_and_destroy(sync_nv12_r);
  wait_and_destroy(sync_bgra_l);
  wait_and_destroy(sync_bgra_r);

  gst_buffer_unmap(buf, &map);  // release NVMM src reference; dst is independent

  // ── CPU-sync the NV12 surfaces so mappedAddr.addr[] is coherent ──────────────
  // NvBufSurfaceSyncForCpu with planeIdx=-1 flushes ALL planes (Y + UV).
  // Using planeIdx=0 only flushes the Y plane; UV retains zero-initialised cache
  // lines → BT.601 matrix yields R≈0, B≈0, G≈Y+135 → green screen.
  // Also sync BGRA surfaces for the visual_stream CPU path.
  NvBufSurfaceSyncForCpu(nvbuf_left_,  0, -1);   // -1 = all planes (Y + UV)
  NvBufSurfaceSyncForCpu(nvbuf_right_, 0, -1);
  if (have_raw_surfs && want_raw) {
    // BGRA is a single packed plane; -1 == 0 here but is more explicit.
    NvBufSurfaceSyncForCpu(nvbuf_raw_left_,  0, -1);
    NvBufSurfaceSyncForCpu(nvbuf_raw_right_, 0, -1);

    // ── GPU lens-distortion rectification (CUDA bilinear remap) ─────────────
    // Reads mappedAddr.addr[0] (now CPU-coherent), remaps via CUDA kernel,
    // writes the rectified pixels back to mappedAddr.addr[0].  Downstream
    // publish_visual_side and make_nitros_image see the rectified BGRA surface.
    if (rectify_left_  && d_rect_map_x_left_)  apply_rect_bgra(nvbuf_raw_left_,  true);
    if (rectify_right_ && d_rect_map_x_right_) apply_rect_bgra(nvbuf_raw_right_, false);
  }

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

  // ── visual_stream publish (rviz2 / network) ──────────────────────────────
  // VIC hw path (preferred): nvbuf_raw_{left,right}_ already contain BGRA data
  // from the async transform above. mappedAddr.addr[0] is directly usable as a
  // cv::Mat — no heap allocation, no memcpy, no cvtColor loop.
  //
  // CPU fallback (if BGRA surfaces weren't allocated): de-stride the NV12
  // SURFACE_ARRAY and run cv::cvtColor — same as the original path.
  //
  // transport="compressed": JPEG-encode BGR/RGB on CPU (~1 ms @ 640×480) and
  // publish sensor_msgs/CompressedImage.  Reduces network bandwidth by ~97%.
  // transport="raw": publish sensor_msgs/Image (raw BGR/RGB, full bandwidth).
  //
  // bgr_cache_*: BGR mat at surface resolution cached here so make_nitros_image
  // can reuse it and skip the second BGRA→BGR NEON pass when both topics active.
  cv::Mat bgr_cache_left, bgr_cache_right;
  auto publish_visual_side = [&](
    NvBufSurface * bgra_surf,   // null → CPU NV12 fallback
    NvBufSurface * nv12_surf,   // used for CPU fallback only
    bool is_left)
  {
    const std::string & transport = is_left ? vis_transport_left_    : vis_transport_right_;
    const std::string & encoding  = is_left ? vis_encoding_left_     : vis_encoding_right_;
    const int           jpeg_q    = is_left ? vis_jpeg_quality_left_ : vis_jpeg_quality_right_;
    const std::string & frame_id  = is_left ? frame_id_left_         : frame_id_right_;
    const int           out_w     = eff_out_w(is_left);
    const int           out_h     = eff_out_h(is_left);

    cv::Mat bgr;
    if (bgra_surf) {
      // ── VIC path: wrap BGRA directly, convert BGRA→BGR (single NEON pass) ──
      auto * bgra_ptr = static_cast<uint8_t *>(
        bgra_surf->surfaceList[0].mappedAddr.addr[0]);
      const size_t pitch = bgra_surf->surfaceList[0].planeParams.pitch[0];
      cv::Mat bgra_mat(combined_height_, ew, CV_8UC4, bgra_ptr, pitch);
      cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);
    } else {
      // ── CPU fallback: de-stride NV12 + cvtColor ────────────────────────────
      auto * y_src  = static_cast<const uint8_t *>(
        nv12_surf->surfaceList[0].mappedAddr.addr[0]);
      auto * uv_src = static_cast<const uint8_t *>(
        nv12_surf->surfaceList[0].mappedAddr.addr[1]);
      const size_t pitch_y  = nv12_surf->surfaceList[0].planeParams.pitch[0];
      const size_t pitch_uv = nv12_surf->surfaceList[0].planeParams.pitch[1];
      // FIX 2.3: use pre-allocated staging buffer instead of per-frame malloc
      std::vector<uint8_t> & nv12_buf = nv12_cpu_staging_;
      // FIX 3.5: single bulk memcpy when VIC output is already packed (pitch == width)
      const size_t y_size  = static_cast<size_t>(ew) * static_cast<size_t>(combined_height_);
      const size_t uv_size = static_cast<size_t>(ew) * static_cast<size_t>(combined_height_ / 2);
      if (pitch_y == static_cast<size_t>(ew)) {
        std::memcpy(nv12_buf.data(), y_src, y_size);
      } else {
        for (int r = 0; r < combined_height_; ++r)
          std::memcpy(nv12_buf.data() + r * ew, y_src + r * pitch_y, static_cast<size_t>(ew));
      }
      if (pitch_uv == static_cast<size_t>(ew)) {
        std::memcpy(nv12_buf.data() + y_size, uv_src, uv_size);
      } else {
        for (int r = 0; r < combined_height_ / 2; ++r)
          std::memcpy(nv12_buf.data() + ew * combined_height_ + r * ew,
                      uv_src + r * pitch_uv, static_cast<size_t>(ew));
      }
      cv::Mat nv12_mat(combined_height_ * 3 / 2, ew, CV_8UC1, nv12_buf.data());
      cv::cvtColor(nv12_mat, bgr, cv::COLOR_YUV2BGR_NV12);
    }

    // Cache at surface resolution so make_nitros_image can reuse it
    (is_left ? bgr_cache_left : bgr_cache_right) = bgr;

    // Apply requested encoding (bgr is already the correct base from cvtColor)
    cv::Mat out;
    if (encoding == "rgb8") {
      cv::cvtColor(bgr, out, cv::COLOR_BGR2RGB);
    } else {
      out = bgr;
    }
    if (out_w != out.cols || out_h != out.rows) {
      cv::resize(out, out, cv::Size(out_w, out_h));
    }

    std_msgs::msg::Header hdr;
    hdr.stamp    = stamp;
    hdr.frame_id = frame_id;

    if (transport == "compressed") {
      // FIX 3.3: JPEG encode off the capture thread (~3–8 ms) so VIC isn't stalled.
      // Wait for previous frame's encode first, clone `out`, then fire async.
      std::future<void> & fut = is_left ? jpeg_future_left_ : jpeg_future_right_;
      if (fut.valid()) fut.get();
      cv::Mat out_copy = out.clone();
      auto pub = is_left ? pub_vis_comp_left_ : pub_vis_comp_right_;
      fut = std::async(std::launch::async,
        [pub, out_copy = std::move(out_copy), enc = encoding, q = jpeg_q, hdr]() mutable {
          std::vector<uchar> jpeg_buf;
          cv::imencode(".jpg", out_copy, jpeg_buf, {cv::IMWRITE_JPEG_QUALITY, q});
          sensor_msgs::msg::CompressedImage msg;
          msg.header = hdr;
          msg.format = enc + "; jpeg compressed " + enc;
          msg.data   = std::move(jpeg_buf);
          pub->publish(msg);
        });
    } else {
      auto img_msg = cv_bridge::CvImage(hdr, encoding, out).toImageMsg();
      (is_left ? pub_vis_raw_left_ : pub_vis_raw_right_)->publish(
        std::make_unique<sensor_msgs::msg::Image>(std::move(*img_msg)));
    }
  };

  if (vis_enable_left_)
    publish_visual_side(have_raw_surfs ? nvbuf_raw_left_  : nullptr, nvbuf_left_,  true);
  if (vis_enable_right_)
    publish_visual_side(have_raw_surfs ? nvbuf_raw_right_ : nullptr, nvbuf_right_, false);

  // ── NITROS publish via TypeAdapter ────────────────────────────────────────
  // Publishing sensor_msgs::msg::Image on rclcpp::Publisher<NitrosImage>
  // triggers convert_to_custom() in the TypeAdapter which creates a
  // lifecycle-managed GXF VideoBuffer (freed after all subscribers consume it).
  //
  // Supported encodings (from params.yaml  left/right_camera.topics.nitros_image.format):
  //   "rgb8"   — packed RGB8  (step = width*3)   recommended for visualisation
  //   "bgr8"   — packed BGR8  (step = width*3)   OpenCV-native byte order
  //   "mono8"  — Y-plane only (step = width)     grayscale
  //   "nv12"   — Y+UV NV12   (step = width)      NOTE: TypeAdapter single-pass
  //              cudaMemcpy2D only copies the Y plane; UV plane in the
  //              GXF VideoBuffer remains uninitialized → downstream sees
  //              random chroma.  Use "rgb8" for reliable colour output.
  //
  // For rgb8/bgr8: VIC already did the heavy NV12→BGRA color matrix in
  // NvBufSurfTransformAsync.  The CPU step is only BGRA→BGR (single NEON
  // channel-drop, ~1 cycle/px).  When publish_visual_side already ran for this
  // side, bgr_cache_{left,right} holds that result at surface resolution so the
  // NEON pass is skipped entirely here.
  // cv::resize handles the optional output-resolution downscale.
  // FIX 2.2: lambda now fills a pre-allocated image in-place — no malloc on hot path.
  // img.data is pre-sized in init_nvbuf_surfaces(); resize() here is a no-op.
  auto make_nitros_image = [&](
    sensor_msgs::msg::Image & img,       // pre-allocated staging image; filled in place
    NvBufSurface * nv12_surf,    // VIC-cropped NV12 surface (eye_width × combined_height)
    NvBufSurface * bgra_surf,    // VIC-cropped BGRA surface, or nullptr
    const cv::Mat & bgr_hint,    // optional BGR mat at surf dims from visual path; skips redo
    const std::string & frame_id,
    const std::string & enc,
    int out_w, int out_h) -> void
  {
    img.header.stamp    = stamp;
    img.header.frame_id = frame_id;
    img.width           = static_cast<uint32_t>(out_w);
    img.height          = static_cast<uint32_t>(out_h);
    img.encoding        = enc;
    img.is_bigendian    = false;

    if (enc == "rgb8" || enc == "bgr8") {
      // ── packed 3-channel path ─────────────────────────────────────────────
      // step = width * 3 so TypeAdapter's cudaMemcpy2D gets
      //   copy_width = get_step_size = width * 3
      //   src_pitch  = img.step     = width * 3
      //   dst_pitch  = GXF stride   = width * 3  (NoPaddingColorPlanes<RGB/BGR>)
      // All equal → always valid.
      img.step = static_cast<uint32_t>(out_w) * 3;
      img.data.resize(static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3);

      const int surf_w = eye_width();
      const int surf_h = combined_height_;
      cv::Mat bgr;
      bool already_rgb = false;  // FIX 3.4: track if bgr already holds RGB data

      if (!bgr_hint.empty()) {
        // Reuse BGR mat already computed by publish_visual_side — no extra NEON pass
        bgr = bgr_hint;
      } else if (bgra_surf) {
        // FIX 3.4: go directly to the target colour space — skip the second pass
        const auto & sl = bgra_surf->surfaceList[0];
        auto * ptr = static_cast<uint8_t *>(sl.mappedAddr.addr[0]);
        cv::Mat bgra_mat(surf_h, surf_w, CV_8UC4, ptr,
                         static_cast<size_t>(sl.planeParams.pitch[0]));
        if (enc == "rgb8") {
          cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2RGB);  // single NEON pass straight to RGB
          already_rgb = true;
        } else {
          cv::cvtColor(bgra_mat, bgr, cv::COLOR_BGRA2BGR);
        }
      } else {
        // CPU NV12 fallback: de-stride then cvtColor
        const auto & sl = nv12_surf->surfaceList[0];
        const auto * y_src  = static_cast<const uint8_t *>(sl.mappedAddr.addr[0]);
        const auto * uv_src = static_cast<const uint8_t *>(sl.mappedAddr.addr[1]);
        const size_t py = sl.planeParams.pitch[0];
        const size_t pu = sl.planeParams.pitch[1];
        // FIX 2.3: use pre-allocated staging buffer instead of per-frame malloc
        std::vector<uint8_t> & nv12_buf = nv12_cpu_staging_;
        // FIX 3.5: single bulk memcpy when VIC output is already packed
        const size_t y_sz  = static_cast<size_t>(surf_w) * static_cast<size_t>(surf_h);
        const size_t uv_sz = static_cast<size_t>(surf_w) * static_cast<size_t>(surf_h / 2);
        if (py == static_cast<size_t>(surf_w)) {
          std::memcpy(nv12_buf.data(), y_src, y_sz);
        } else {
          for (int r = 0; r < surf_h; ++r)
            std::memcpy(nv12_buf.data() + r * surf_w, y_src + r * py, surf_w);
        }
        if (pu == static_cast<size_t>(surf_w)) {
          std::memcpy(nv12_buf.data() + y_sz, uv_src, uv_sz);
        } else {
          for (int r = 0; r < surf_h / 2; ++r)
            std::memcpy(nv12_buf.data() + surf_w * surf_h + r * surf_w,
                        uv_src + r * pu, surf_w);
        }
        cv::Mat nv12_mat(surf_h * 3 / 2, surf_w, CV_8UC1, nv12_buf.data());
        cv::cvtColor(nv12_mat, bgr, cv::COLOR_YUV2BGR_NV12);
      }

      // Resize to requested NITROS output resolution if needed
      if (out_w != bgr.cols || out_h != bgr.rows)
        cv::resize(bgr, bgr, cv::Size(out_w, out_h));

      // Convert BGR → RGB if needed (skipped when FIX 3.4 already produced RGB above)
      cv::Mat * out_ptr = &bgr;
      cv::Mat rgb_tmp;
      if (enc == "rgb8" && !already_rgb) {
        cv::cvtColor(bgr, rgb_tmp, cv::COLOR_BGR2RGB);
        out_ptr = &rgb_tmp;
      }

      // step must equal out_w*3 for packed layout (cv::resize output is always packed)
      std::memcpy(img.data.data(), out_ptr->data,
                  static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3);

    } else if (enc == "mono8") {
      // ── single-plane luminance path ───────────────────────────────────────
      img.step = static_cast<uint32_t>(out_w);
      img.data.resize(static_cast<size_t>(out_w) * static_cast<size_t>(out_h));
      const auto & sl = nv12_surf->surfaceList[0];
      const auto * y  = static_cast<const uint8_t *>(sl.mappedAddr.addr[0]);
      const size_t py = sl.planeParams.pitch[0];
      // FIX 3.5: bulk copy when VIC Y plane is packed
      if (py == static_cast<size_t>(out_w)) {
        std::memcpy(img.data.data(), y,
                    static_cast<size_t>(out_w) * static_cast<size_t>(out_h));
      } else {
        for (int r = 0; r < out_h; ++r)
          std::memcpy(img.data.data() + r * out_w, y + r * py,
                      static_cast<size_t>(out_w));
      }

    } else {
      // ── NV12 (and any unrecognised encoding) ──────────────────────────────
      // TypeAdapter convert_to_custom calls a SINGLE cudaMemcpy2D with
      //   copy_width = get_step_size = width * bpp_Y = width * 1 = width
      //   height     = img.height
      // This copies only the Y plane (width*height bytes).  The UV plane in
      // the GXF VideoBuffer is never touched and remains uninitialized.
      // For correctly coloured NITROS output, set format = "rgb8" instead.
      if (enc != "nv12") {
        RCLCPP_WARN_ONCE(get_logger(),
          "make_nitros_image: unrecognised encoding '%s' — treating as nv12.  "
          "Supported: rgb8, bgr8, mono8, nv12.", enc.c_str());
      }
      img.step = static_cast<uint32_t>(out_w);
      img.data.resize(static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * 3 / 2);
      uint8_t * dst = img.data.data();
      const auto & sl = nv12_surf->surfaceList[0];
      const auto * y   = static_cast<const uint8_t *>(sl.mappedAddr.addr[0]);
      const auto * uv  = static_cast<const uint8_t *>(sl.mappedAddr.addr[1]);
      const size_t py  = sl.planeParams.pitch[0];
      const size_t pu  = sl.planeParams.pitch[1];
      // FIX 3.5: bulk copy when VIC Y/UV planes are packed
      const size_t y_bytes  = static_cast<size_t>(out_w) * static_cast<size_t>(out_h);
      const size_t uv_bytes = static_cast<size_t>(out_w) * static_cast<size_t>(out_h / 2);
      if (py == static_cast<size_t>(out_w)) {
        std::memcpy(dst, y, y_bytes);
      } else {
        for (int r = 0; r < out_h; ++r)
          std::memcpy(dst + r * out_w, y + r * py, static_cast<size_t>(out_w));
      }
      uint8_t * dst_uv = dst + out_w * out_h;
      if (pu == static_cast<size_t>(out_w)) {
        std::memcpy(dst_uv, uv, uv_bytes);
      } else {
        for (int r = 0; r < out_h / 2; ++r)
          std::memcpy(dst_uv + r * out_w, uv + r * pu, static_cast<size_t>(out_w));
      }
    }
    // void lambda — img is modified in place; no return value
  };

  RCLCPP_INFO_ONCE(get_logger(),
    "NITROS publish via TypeAdapter  left='%s'  right='%s'  eye=%dx%d",
    nitros_fmt_left_.c_str(), nitros_fmt_right_.c_str(), ew, combined_height_);

  if (pub_nitros_left_) {
    make_nitros_image(
      nitros_img_left_,
      nvbuf_left_,  have_raw_surfs ? nvbuf_raw_left_  : nullptr,
      bgr_cache_left,
      frame_id_left_,  nitros_fmt_left_,
      eff_out_w_nitros(true),  eff_out_h_nitros(true));
    pub_left_nitros_->publish(nitros_img_left_);
  }
  if (pub_nitros_right_) {
    make_nitros_image(
      nitros_img_right_,
      nvbuf_right_, have_raw_surfs ? nvbuf_raw_right_ : nullptr,
      bgr_cache_right,
      frame_id_right_, nitros_fmt_right_,
      eff_out_w_nitros(false), eff_out_h_nitros(false));
    pub_right_nitros_->publish(nitros_img_right_);
  }

  return true;
}

#endif  // HAVE_NVBUF

// ─────────────────────────────────────────────────────────────────────────────
// capture_loop()  — runs in a dedicated thread
// ─────────────────────────────────────────────────────────────────────────────
void ArducamB0573Node::capture_loop()
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
void ArducamB0573Node::process_sample(GstSample * sample)
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

  // visual_stream published per-side only when visual_stream.enable is true
  auto publish_cpu_visual = [&](
    cv::Mat & bgr,
    bool is_left)
  {
    const std::string & transport = is_left ? vis_transport_left_    : vis_transport_right_;
    const std::string & encoding  = is_left ? vis_encoding_left_     : vis_encoding_right_;
    const int           jpeg_q    = is_left ? vis_jpeg_quality_left_ : vis_jpeg_quality_right_;
    const std::string & frame_id  = is_left ? frame_id_left_         : frame_id_right_;
    const int           out_w     = eff_out_w(is_left);
    const int           out_h     = eff_out_h(is_left);

    cv::Mat out;
    if (encoding == "rgb8") {
      cv::cvtColor(bgr, out, cv::COLOR_BGR2RGB);
    } else {
      out = bgr;  // bgr8 default — zero copy
    }
    if (out_w != out.cols || out_h != out.rows) {
      cv::resize(out, out, cv::Size(out_w, out_h));
    }

    std_msgs::msg::Header hdr;
    hdr.stamp    = stamp;
    hdr.frame_id = frame_id;

    if (transport == "compressed") {
      // FIX 3.3 (CPU path): same async JPEG pattern as NVBUF path.
      std::future<void> & fut = is_left ? jpeg_future_left_ : jpeg_future_right_;
      if (fut.valid()) fut.get();
      cv::Mat out_copy = out.clone();
      auto pub = is_left ? pub_vis_comp_left_ : pub_vis_comp_right_;
      fut = std::async(std::launch::async,
        [pub, out_copy = std::move(out_copy), enc = encoding, q = jpeg_q, hdr]() mutable {
          std::vector<uchar> jpeg_buf;
          cv::imencode(".jpg", out_copy, jpeg_buf, {cv::IMWRITE_JPEG_QUALITY, q});
          sensor_msgs::msg::CompressedImage msg;
          msg.header = hdr;
          msg.format = enc + "; jpeg compressed " + enc;
          msg.data   = std::move(jpeg_buf);
          pub->publish(msg);
        });
    } else {
      auto sp = cv_bridge::CvImage(hdr, encoding, out).toImageMsg();
      (is_left ? pub_vis_raw_left_ : pub_vis_raw_right_)->publish(
        std::make_unique<sensor_msgs::msg::Image>(std::move(*sp)));
    }
  };

  if (vis_enable_left_)  publish_cpu_visual(left_bgr,  true);
  if (vis_enable_right_) publish_cpu_visual(right_bgr, false);
}

}  // namespace arducam
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::arducam::ArducamB0573Node)
