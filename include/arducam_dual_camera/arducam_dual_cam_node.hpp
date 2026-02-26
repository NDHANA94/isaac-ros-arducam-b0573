#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>

namespace arducam_dual_camera
{

class ArducamDualCamNode : public rclcpp::Node
{
public:
  explicit ArducamDualCamNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ArducamDualCamNode() override;

private:
  // ── Parameters ───────────────────────────────────────────────────────────
  std::string device_;            // e.g. /dev/video0
  int combined_width_;            // full side-by-side width  (e.g. 2560)
  int combined_height_;           // full height              (e.g. 720)
  int fps_;                       // 0 = let driver decide
  std::string pixel_format_;      // UYVY or NV16
  std::string frame_id_left_;
  std::string frame_id_right_;
  std::string camera_namespace_;  // topic prefix, e.g. "arducam"
  std::string camera_name_left_;  // used by camera_info_manager
  std::string camera_name_right_;
  std::string calib_url_left_;    // file:///path/to/left.yaml  or ""
  std::string calib_url_right_;
  std::string qos_reliability_;   // "best_effort" | "reliable"
  std::string qos_durability_;    // "volatile"    | "transient_local"

  // ── GStreamer ─────────────────────────────────────────────────────────────
  GstElement * pipeline_{nullptr};
  GstElement * appsink_{nullptr};
  std::thread  capture_thread_;
  std::atomic<bool> running_{false};

  void build_pipeline();
  void capture_loop();
  void process_sample(GstSample * sample);

  // ── Publishers ────────────────────────────────────────────────────────────
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      pub_left_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      pub_right_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_left_info_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_right_info_;

  // ── Camera info managers ──────────────────────────────────────────────────
  std::shared_ptr<camera_info_manager::CameraInfoManager> cim_left_;
  std::shared_ptr<camera_info_manager::CameraInfoManager> cim_right_;
};

}  // namespace arducam_dual_camera
