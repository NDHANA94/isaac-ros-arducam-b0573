/**
 * arducam_dual_cam_node.cpp
 *
 * Single ROS 2 node for the Arducam B0573 (GMSL2-to-CSI2) dual camera.
 * Hardware pipeline:
 *   v4l2src  →  nvvidconv (Jetson VIC)  →  appsink (BGRx, CPU memory)
 *
 * The combined side-by-side frame is split at width/2 into left and right
 * halves using a cv::Mat ROI (shallow, no pixel copy) and published as
 * sensor_msgs/Image on:
 *   /arducam/left/image_raw    /arducam/left/camera_info
 *   /arducam/right/image_raw   /arducam/right/camera_info
 *
 * Parameters:
 *   device              (string)  /dev/video0
 *   width               (int)     2560   ← combined width (1280 per eye)
 *   height              (int)     720
 *   fps                 (int)     30     (0 = let driver negotiate)
 *   pixel_format        (string)  UYVY
 *   frame_id_left       (string)  left_camera
 *   frame_id_right      (string)  right_camera
 *   camera_namespace    (string)  arducam       ← topic prefix
 *   camera_name_left    (string)  left          ← used by camera_info_manager
 *   camera_name_right   (string)  right
 *   calib_url_left      (string)  ""            ← empty = identity/uncalibrated
 *   calib_url_right     (string)  ""            ← e.g. file:///path/to/right.yaml
 *   qos.reliability     (string)  best_effort   ← "best_effort" | "reliable"
 *   qos.durability      (string)  volatile      ← "volatile" | "transient_local"
 *
 * Calibration file format: standard ROS camera_info YAML
 *   (produced by `ros2 run camera_calibration cameracalibrator`)
 *   URL examples:
 *     file:///home/orin/calibration/left.yaml
 *     package://arducam_dual_camera/config/left.yaml
 *
 * Available resolutions (Arducam B0573 on /dev/video0):
 *   3840×1200  →  1920×1200 per eye
 *   2560×720   →  1280×720  per eye  (default)
 *   1280×480   →   640×480  per eye
 */

#include "arducam_dual_camera/arducam_dual_cam_node.hpp"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>   // GstVideoInfo — needed for stride query

namespace arducam_dual_camera
{

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
ArducamDualCamNode::ArducamDualCamNode(const rclcpp::NodeOptions & options)
: Node("arducam_dual_cam_node", options)
{
  // ── Declare parameters ──────────────────────────────────────────────────
  declare_parameter("device",             "/dev/video0");
  declare_parameter("width",              2560);
  declare_parameter("height",             720);
  declare_parameter("fps",               30);
  declare_parameter("pixel_format",       "UYVY");
  declare_parameter("frame_id_left",      "left_camera");
  declare_parameter("frame_id_right",     "right_camera");
  declare_parameter("camera_namespace",   "arducam");
  declare_parameter("camera_name_left",   "left");
  declare_parameter("camera_name_right",  "right");
  declare_parameter("calib_url_left",     "");
  declare_parameter("calib_url_right",    "");
  declare_parameter("qos.reliability",    "best_effort");
  declare_parameter("qos.durability",     "volatile");

  device_              = get_parameter("device").as_string();
  combined_width_      = get_parameter("width").as_int();
  combined_height_     = get_parameter("height").as_int();
  fps_                 = get_parameter("fps").as_int();
  pixel_format_        = get_parameter("pixel_format").as_string();
  frame_id_left_       = get_parameter("frame_id_left").as_string();
  frame_id_right_      = get_parameter("frame_id_right").as_string();
  camera_namespace_    = get_parameter("camera_namespace").as_string();
  camera_name_left_    = get_parameter("camera_name_left").as_string();
  camera_name_right_   = get_parameter("camera_name_right").as_string();
  calib_url_left_      = get_parameter("calib_url_left").as_string();
  calib_url_right_     = get_parameter("calib_url_right").as_string();
  qos_reliability_     = get_parameter("qos.reliability").as_string();
  qos_durability_      = get_parameter("qos.durability").as_string();

  RCLCPP_INFO(get_logger(),
    "Arducam B0573 | device=%s  combined=%dx%d  fps=%d  fmt=%s",
    device_.c_str(), combined_width_, combined_height_, fps_, pixel_format_.c_str());
  RCLCPP_INFO(get_logger(),
    "Per-eye resolution: %dx%d  namespace=%s  qos=[%s/%s]",
    combined_width_ / 2, combined_height_,
    camera_namespace_.c_str(), qos_reliability_.c_str(), qos_durability_.c_str());

  // ── Publishers ──────────────────────────────────────────────────────────
  // Build QoS from parameters
  rclcpp::QoS img_qos(rclcpp::KeepLast(10));

  if (qos_reliability_ == "reliable") {
    img_qos.reliable();
  } else {
    img_qos.best_effort();
  }

  if (qos_durability_ == "transient_local") {
    img_qos.transient_local();
  } else {
    img_qos.durability_volatile();
  }

  // Topic names derived from camera_namespace parameter
  const std::string ns         = camera_namespace_;
  const std::string left_img   = ns + "/left/image_raw";
  const std::string right_img  = ns + "/right/image_raw";
  const std::string left_info  = ns + "/left/camera_info";
  const std::string right_info = ns + "/right/camera_info";

  RCLCPP_INFO(get_logger(), "Publishing on:");
  RCLCPP_INFO(get_logger(), "  %s", left_img.c_str());
  RCLCPP_INFO(get_logger(), "  %s", right_img.c_str());
  RCLCPP_INFO(get_logger(), "  %s", left_info.c_str());
  RCLCPP_INFO(get_logger(), "  %s", right_info.c_str());

  pub_left_       = create_publisher<sensor_msgs::msg::Image>(left_img,   img_qos);
  pub_right_      = create_publisher<sensor_msgs::msg::Image>(right_img,  img_qos);
  pub_left_info_  = create_publisher<sensor_msgs::msg::CameraInfo>(left_info,  img_qos);
  pub_right_info_ = create_publisher<sensor_msgs::msg::CameraInfo>(right_info, img_qos);

  // ── Camera info managers ────────────────────────────────────────────────
  // camera_info_manager loads the YAML calibration file at the given URL.
  // If the URL is empty or the file is missing it returns an identity/zeroed
  // CameraInfo and logs a warning — the node keeps running uncalibrated.
  cim_left_  = std::make_shared<camera_info_manager::CameraInfoManager>(
    this, camera_name_left_,  calib_url_left_);
  cim_right_ = std::make_shared<camera_info_manager::CameraInfoManager>(
    this, camera_name_right_, calib_url_right_);

  if (calib_url_left_.empty()) {
    RCLCPP_WARN(get_logger(),
      "calib_url_left is empty — publishing uncalibrated left camera_info");
  } else {
    RCLCPP_INFO(get_logger(), "Left  calibration: %s", calib_url_left_.c_str());
  }
  if (calib_url_right_.empty()) {
    RCLCPP_WARN(get_logger(),
      "calib_url_right is empty — publishing uncalibrated right camera_info");
  } else {
    RCLCPP_INFO(get_logger(), "Right calibration: %s", calib_url_right_.c_str());
  }

  // ── GStreamer ────────────────────────────────────────────────────────────
  int   argc = 0;
  char ** argv = nullptr;
  gst_init(&argc, &argv);

  build_pipeline();

  running_ = true;
  capture_thread_ = std::thread(&ArducamDualCamNode::capture_loop, this);
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
ArducamDualCamNode::~ArducamDualCamNode()
{
  running_ = false;

  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);

    // Unblock any blocking pull_sample by sending EOS
    GstElement * src = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
    if (src) {
      gst_element_send_event(src, gst_event_new_eos());
      gst_object_unref(src);
    }
  }

  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }

  if (pipeline_) {
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// build_pipeline()
// ─────────────────────────────────────────────────────────────────────────────
void ArducamDualCamNode::build_pipeline()
{
  // nvvidconv: Jetson VIC engine — converts UYVY → BGRx in hardware
  // appsink: blocking pull_sample, max 2 buffers, drops old frames when full
  std::string fps_caps = (fps_ > 0)
    ? (std::string(",framerate=") + std::to_string(fps_) + "/1")
    : "";

  std::string pipeline_str =
    "v4l2src name=src device=" + device_ +
    " ! video/x-raw,format=" + pixel_format_ +
    ",width="  + std::to_string(combined_width_) +
    ",height=" + std::to_string(combined_height_) +
    fps_caps +
    " ! nvvidconv"
    " ! video/x-raw,format=BGRx"
    " ! appsink name=sink sync=false max-buffers=2 drop=true";

  RCLCPP_INFO(get_logger(), "GStreamer pipeline: %s", pipeline_str.c_str());

  GError * error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);

  if (!pipeline_ || error) {
    RCLCPP_FATAL(get_logger(), "Failed to build GStreamer pipeline: %s",
      error ? error->message : "(unknown)");
    if (error) g_error_free(error);
    rclcpp::shutdown();
    return;
  }

  appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!appsink_) {
    RCLCPP_FATAL(get_logger(), "Could not find appsink element");
    rclcpp::shutdown();
    return;
  }

  // Configure appsink: emit signals=false, we use blocking pull_sample
  gst_app_sink_set_emit_signals(GST_APP_SINK(appsink_), FALSE);

  GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    RCLCPP_FATAL(get_logger(), "Failed to start GStreamer pipeline");
    rclcpp::shutdown();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// capture_loop()  — runs in a dedicated thread
// ─────────────────────────────────────────────────────────────────────────────
void ArducamDualCamNode::capture_loop()
{
  RCLCPP_INFO(get_logger(), "Capture thread started");

  while (running_) {
    // Blocking pull — returns nullptr on EOS or error
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

  // ── Determine the actual row stride from GStreamer video caps ─────────────
  // nvvidconv on Jetson aligns DMA buffer rows (often to 64 or 128 bytes).
  // If we assume stride = width*4 and it is actually larger, every row is
  // offset and the Mat is corrupt → causes "sequence size exceeds remaining
  // buffer" during ROS serialization.  Always read the real stride.
  GstCaps * caps = gst_sample_get_caps(sample);
  GstVideoInfo vinfo;
  gst_video_info_init(&vinfo);
  size_t actual_stride = static_cast<size_t>(combined_width_) * 4;  // fallback
  if (caps && gst_video_info_from_caps(&vinfo, caps)) {
    actual_stride = static_cast<size_t>(GST_VIDEO_INFO_PLANE_STRIDE(&vinfo, 0));
    RCLCPP_INFO_ONCE(get_logger(),
      "GStreamer video info: %dx%d  stride=%zu  (nominal row bytes=%d)",
      GST_VIDEO_INFO_WIDTH(&vinfo), GST_VIDEO_INFO_HEIGHT(&vinfo),
      actual_stride, combined_width_ * 4);
  } else {
    RCLCPP_WARN_ONCE(get_logger(),
      "Could not parse GstVideoInfo from caps; assuming packed stride %zu",
      actual_stride);
  }

  GstMapInfo map{};
  if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
    RCLCPP_WARN(get_logger(), "Failed to map GStreamer buffer");
    return;
  }

  // Minimum expected bytes: stride may be >= width*4
  const size_t min_expected =
    actual_stride * static_cast<size_t>(combined_height_ - 1)
    + static_cast<size_t>(combined_width_) * 4;
  if (map.size < min_expected) {
    RCLCPP_WARN_ONCE(get_logger(),
      "Buffer size mismatch: got %zu, expected >= %zu  (stride=%zu)",
      map.size, min_expected, actual_stride);
    gst_buffer_unmap(buf, &map);
    return;
  }

  // Wrap the DMA buffer as a cv::Mat with the REAL stride — no pixel copy
  cv::Mat combined(combined_height_, combined_width_, CV_8UC4,
                   map.data, actual_stride);

  // Split left / right via ROI — still no pixel copy
  int half_w = combined_width_ / 2;
  cv::Mat left_bgra  = combined(cv::Rect(0,      0, half_w, combined_height_));
  cv::Mat right_bgra = combined(cv::Rect(half_w, 0, half_w, combined_height_));

  // Convert BGRx → BGR8  (fast channel drop, single pass)
  // cvtColor output is a new, contiguous, owned Mat — safe to publish after unmap
  cv::Mat left_bgr, right_bgr;
  cv::cvtColor(left_bgra,  left_bgr,  cv::COLOR_BGRA2BGR);
  cv::cvtColor(right_bgra, right_bgr, cv::COLOR_BGRA2BGR);

  gst_buffer_unmap(buf, &map);

  // ── Build ROS timestamp ───────────────────────────────────────────────────
  rclcpp::Time stamp = now();

  // ── Publish images ────────────────────────────────────────────────────────
  // Left
  {
    auto msg = cv_bridge::CvImage(
      std_msgs::msg::Header{}, "bgr8", left_bgr).toImageMsg();
    msg->header.stamp    = stamp;
    msg->header.frame_id = frame_id_left_;
    pub_left_->publish(std::move(*msg));
  }

  // Right
  {
    auto msg = cv_bridge::CvImage(
      std_msgs::msg::Header{}, "bgr8", right_bgr).toImageMsg();
    msg->header.stamp    = stamp;
    msg->header.frame_id = frame_id_right_;
    pub_right_->publish(std::move(*msg));
  }

  // ── Publish camera_info ───────────────────────────────────────────────────
  // camera_info_manager returns the loaded calibration (or identity if none).
  // We override header.stamp and header.frame_id to match this frame.
  auto info_left  = cim_left_->getCameraInfo();
  auto info_right = cim_right_->getCameraInfo();

  info_left.header.stamp    = stamp;
  info_left.header.frame_id = frame_id_left_;
  info_right.header.stamp    = stamp;
  info_right.header.frame_id = frame_id_right_;

  pub_left_info_->publish(info_left);
  pub_right_info_->publish(info_right);
}

}  // namespace arducam_dual_camera

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<arducam_dual_camera::ArducamDualCamNode>());
  rclcpp::shutdown();
  return 0;
}
