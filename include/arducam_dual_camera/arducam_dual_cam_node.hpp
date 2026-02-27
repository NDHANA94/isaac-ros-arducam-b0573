/**
 * @file arducam_dual_cam_node.hpp
 * @author WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
 * @date 26.02.2026 
*/


#ifndef ARDUCAM_DUAL_CAMERA__ARDUCAM_DUAL_CAM_NODE_HPP_
#define ARDUCAM_DUAL_CAMERA__ARDUCAM_DUAL_CAM_NODE_HPP_

#pragma once

#include <atomic>
#include <ctime>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2/LinearMath/Quaternion.h>

// NvBufSurface/NvBufSurfTransform: VIC-native zero-copy NVMM API (DeepStream 7)
// Isaac ROS NITROS: TypeAdapter path — publish sensor_msgs::msg::Image on a
// rclcpp::Publisher<NitrosImage>; convert_to_custom() creates a lifecycle-managed
// GXF VideoBuffer without any cudaMalloc or NitrosImageBuilder involvement.
#ifdef HAVE_NVBUF
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
// NitrosImage TypeAdapter: rclcpp::TypeAdapter<NitrosImage, sensor_msgs::msg::Image>
// Publishing sensor_msgs::msg::Image on Publisher<NitrosImage> triggers
// convert_to_custom() which allocates and frees the GXF VideoBuffer correctly.
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#endif

namespace arducam_dual_camera
{

class ArducamDualCamNode : public rclcpp::Node
{
public:
  explicit ArducamDualCamNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ArducamDualCamNode() override;

private:
  // ── Global parameters ─────────────────────────────────────────────────────
  std::string device_;
  int combined_width_;      // full side-by-side width (e.g. 2560)
  int combined_height_;     // full height              (e.g. 720)
  int fps_;
  std::string pixel_format_;

  // ── Per-camera configuration (left_camera.* / right_camera.* params) ───────
  std::string topic_vis_left_;       // resolved visual_stream topic (<prefix>/image_raw or /image/compressed)
  std::string topic_vis_right_;
  std::string topic_left_info_;      // full camera_info topic name
  std::string topic_right_info_;
  std::string frame_id_left_;
  std::string frame_id_right_;
  // visual_stream output resolution (-1 = eye_width() / combined_height_)
  int         out_w_vis_left_{-1};
  int         out_h_vis_left_{-1};
  int         out_w_vis_right_{-1};
  int         out_h_vis_right_{-1};
  // nitros_image output resolution (-1 = eye_width() / combined_height_)
  int         out_w_nitros_left_{-1};
  int         out_h_nitros_left_{-1};
  int         out_w_nitros_right_{-1};
  int         out_h_nitros_right_{-1};

  // Pre-built CameraInfo (constructed once in the constructor from inline intrinsic params)
  sensor_msgs::msg::CameraInfo cam_info_left_;
  sensor_msgs::msg::CameraInfo cam_info_right_;

  // Per-camera, per-topic QoS (vis / info / nitros each settable independently)
  std::string qos_vis_rel_left_{"best_effort"},  qos_vis_dur_left_{"volatile"};
  std::string qos_vis_rel_right_{"best_effort"}, qos_vis_dur_right_{"volatile"};
  std::string qos_info_rel_left_{"reliable"},    qos_info_dur_left_{"volatile"};
  std::string qos_info_rel_right_{"reliable"},   qos_info_dur_right_{"volatile"};
  std::string qos_nitros_rel_left_{"best_effort"},  qos_nitros_dur_left_{"volatile"};
  std::string qos_nitros_rel_right_{"best_effort"}, qos_nitros_dur_right_{"volatile"};

  // Per-camera publish control (from left_camera/right_camera params)
  bool        vis_enable_left_{true};          // left_camera.topics.visual_stream.enable
  bool        vis_enable_right_{true};         // right_camera.topics.visual_stream.enable
  bool        pub_nitros_left_{true};          // left_camera.topics.nitros_image.enable
  bool        pub_nitros_right_{true};         // right_camera.topics.nitros_image.enable
  std::string nitros_fmt_left_{"nv12"};       // left_camera.topics.nitros_image.format
  std::string nitros_fmt_right_{"nv12"};      // right_camera.topics.nitros_image.format
  // visual_stream transport and encoding
  std::string vis_transport_left_{"compressed"};  // "raw" | "compressed"
  std::string vis_transport_right_{"compressed"};
  std::string vis_encoding_left_{"bgr8"};          // "bgr8" | "rgb8"
  std::string vis_encoding_right_{"bgr8"};
  int         vis_jpeg_quality_left_{80};
  int         vis_jpeg_quality_right_{80};

  // static TF broadcaster
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

  // FIX 3.3: JPEG encode futures — encode runs off the capture thread so VIC
  // isn't stalled by libjpeg (3–8 ms @ 640×480). Shared by NVBUF and CPU paths.
  std::future<void> jpeg_future_left_;
  std::future<void> jpeg_future_right_;

  // ── GStreamer ─────────────────────────────────────────────────────────────
  // Preferred pipeline (NVMM→NV12):
  //   v4l2src(NVMM,UYVY) → nvvidconv/VIC → NV12(NVMM) → appsink
  //   gst_buffer_map() returns NvBufSurface* — zero CPU pixel copy.
  //   NvBufSurfTransformAsync() crops left/right halves and converts to BGR
  //   via VIC in hardware; CPU only runs NvBufSurface API calls.
  //
  // Fallback pipelines (tried in order):
  //   NVMM BGRx  → NvBufSurfTransform crop+BGR  (VIC path, slightly more DMA)
  //   System BGRx → CPU ROI + cv::cvtColor       (no NVMM)
  GstElement *      pipeline_{nullptr};
  GstElement *      appsink_{nullptr};
  std::thread       capture_thread_;
  std::atomic<bool> running_{false};

  bool        use_nvmm_{false};       // true → pipeline negotiated memory:NVMM
  std::string gst_out_fmt_{"BGRx"};   // format negotiated for nvvidconv output
  int64_t     gst_clock_offset_{0};   // ns: CLOCK_REALTIME − CLOCK_MONOTONIC

  int eye_width() const { return combined_width_ / 2; }

  // Effective published dimensions for visual_stream
  int eff_out_w(bool left) const {
    int v = left ? out_w_vis_left_ : out_w_vis_right_;
    return (v > 0) ? v : eye_width();
  }
  int eff_out_h(bool left) const {
    int v = left ? out_h_vis_left_ : out_h_vis_right_;
    return (v > 0) ? v : combined_height_;
  }
  // Effective published dimensions for nitros_image
  int eff_out_w_nitros(bool left) const {
    int v = left ? out_w_nitros_left_ : out_w_nitros_right_;
    return (v > 0) ? v : eye_width();
  }
  int eff_out_h_nitros(bool left) const {
    int v = left ? out_h_nitros_left_ : out_h_nitros_right_;
    return (v > 0) ? v : combined_height_;
  }

  // try_build_pipeline: attempts to set the pipeline to PLAYING.
  // nvmm  – request memory:NVMM caps from v4l2src and nvvidconv
  // out_fmt – "NV12" or "BGRx"; the nvvidconv output format
  bool try_build_pipeline(bool nvmm, const std::string & out_fmt = "BGRx");
  void build_pipeline();
  void capture_loop();
  void process_sample(GstSample * sample);

  // ── Publishers ────────────────────────────────────────────────────────────
  // visual_stream publishers — one of raw or compressed per side, set by transport param.
  // nullptr when disabled (vis_enable_left_/right_=false).
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           pub_vis_raw_left_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           pub_vis_raw_right_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_vis_comp_left_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_vis_comp_right_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr      pub_left_info_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr      pub_right_info_;

  // NITROS publishers: GPU-resident zero-copy topic.
  // Full topic names always declared so the constructor log can reference them.
  // Populated from {left,right}_camera.topics.nitros_image.topic_name param.
  std::string topic_left_nitros_;
  std::string topic_right_nitros_;
  // Publishers created only when HAVE_NVBUF is compiled in.
#ifdef HAVE_NVBUF
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr pub_left_nitros_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr pub_right_nitros_;
#endif

  // ── Private helpers ───────────────────────────────────────────────────────
  /// Build a sensor_msgs::CameraInfo from the inline intrinsic params for the given side.
  sensor_msgs::msg::CameraInfo build_cam_info(const std::string & side, int w, int h);
  /// Broadcast a static TF transform (extrinsics) for one camera side.
  void broadcast_static_tf(const std::string & side, const std::string & child_frame);


  // ── NvBufSurface / VIC hardware path ─────────────────────────────────────
  // Available when HAVE_NVBUF is defined and use_nvmm_ is true.
  //
  // nvbuf_left_ / nvbuf_right_:
  //   Pre-allocated, permanently CPU-mapped destination NvBufSurfaces (BGR24).
  //   VIC writes cropped BGR pixels into these via NvBufSurfTransformAsync().
  //   Allocated once in init_nvbuf_surfaces(); destroyed in cleanup_nvbuf_surfaces().
  //
  // process_sample_nvbuf():
  //   1. gst_buffer_map(NVMM) → NvBufSurface* src  [O(1), no pixel copy]
  //   2. NvBufSurfTransform (sync) ×2 → VIC NV12 crop per eye
  //   3. NvBufSurfTransformAsync ×2 → VIC NV12→BGRA crop per eye (visual_stream)
  //   4. Wait for BGRA VIC sync objects.  gst_buffer_unmap.
  //   5. NvBufSurfaceSyncForCpu (all planes) ×4 — flush CPU cache coherency.
  //   6. Build sensor_msgs::msg::Image: std::memcpy NV12 from mappedAddr (stride-strip).
  //   7. pub_{left,right}_nitros_->publish(ros_img)  → TypeAdapter convert_to_custom()
  //      allocates a lifecycle-managed GXF VideoBuffer; freed after consumption.
#ifdef HAVE_NVBUF
  // GXF shared context is managed transparently by the NitrosImage TypeAdapter.
  // We do NOT hold a NitrosContext here — explicit context management caused the
  // 16-entity GXF pool exhaustion (NitrosImageBuilder::Build() leaked one entity
  // per call; 2 sides × 8 frames = 16 = pool full = SIGSEGV).

  bool           use_nvbuf_{false};       // true once surfaces allocated successfully

  // FIX 3.2: persistent VIC session — required for NvBufSurfTransformAsync to
  // actually submit work (without it the call silently does nothing on Orin).
  NvBufSurfTransformConfigParams vic_session_{};

  // NV12 SURFACE_ARRAY VIC dst — used for NITROS publish path.
  NvBufSurface * nvbuf_left_{nullptr};
  NvBufSurface * nvbuf_right_{nullptr};

  // BGRA SURFACE_ARRAY VIC dst — used for image_raw (rviz2) publish path.
  NvBufSurface * nvbuf_raw_left_{nullptr};
  NvBufSurface * nvbuf_raw_right_{nullptr};

  // FIX 2.2: Pre-allocated staging images — reused every frame to avoid 120×/s malloc.
  sensor_msgs::msg::Image nitros_img_left_;
  sensor_msgs::msg::Image nitros_img_right_;

  // FIX 2.3: Shared NV12 CPU de-stride buffer — sized eye_w × combined_h × 3/2.
  std::vector<uint8_t> nv12_cpu_staging_;

  // Allocate NV12 CUDA-device NvBufSurfaces per eye; set use_nvbuf_ = true.
  void init_nvbuf_surfaces();
  void cleanup_nvbuf_surfaces();

  // Returns true if the VIC path completed successfully; false causes
  // process_sample() to fall through to the CPU path.
  bool process_sample_nvbuf(GstBuffer * buf, const rclcpp::Time & stamp);
#endif  // HAVE_NVBUF
};

}  // namespace arducam_dual_camera


#endif  // ARDUCAM_DUAL_CAMERA__ARDUCAM_DUAL_CAM_NODE_HPP_