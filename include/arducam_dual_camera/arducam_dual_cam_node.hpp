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
#include <memory>
#include <string>
#include <thread>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2/LinearMath/Quaternion.h>

// NvBufSurface/NvBufSurfTransform: VIC-native zero-copy NVMM API (DeepStream 7)
// Isaac ROS NITROS: GPU-resident NvBufSurface → zero-copy ROS transport
#ifdef HAVE_NVBUF
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <cuda_runtime_api.h>
// NitrosImage: wraps a CUDA device-memory buffer in a GXF VideoBuffer.
// A co-located NITROS subscriber (same process) receives the GPU handle
// without any host DMA; inter-process subscribers get auto-serialised.
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_builder.hpp"
#include "isaac_ros_nitros/nitros_context.hpp"
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

  // ── Per-camera configuration (cam_left.* / cam_right.* params) ───────────
  std::string topic_left_;        // topic prefix, e.g. /arducam/left  (stored as full image_raw)
  std::string topic_right_;
  std::string topic_left_info_;   // = prefix + "/camera_info"
  std::string topic_right_info_;
  std::string frame_id_left_;
  std::string frame_id_right_;
  int         out_w_left_{-1};    // published image width;  -1 = eye_width()
  int         out_h_left_{-1};    // published image height; -1 = combined_height_
  int         out_w_right_{-1};
  int         out_h_right_{-1};

  // Pre-built CameraInfo (constructed once in the constructor from inline intrinsic params)
  sensor_msgs::msg::CameraInfo cam_info_left_;
  sensor_msgs::msg::CameraInfo cam_info_right_;

  // Per-camera, per-topic QoS (raw / info / nitros each settable independently)
  std::string qos_raw_rel_left_{"reliable"},    qos_raw_dur_left_{"volatile"};
  std::string qos_raw_rel_right_{"reliable"},   qos_raw_dur_right_{"volatile"};
  std::string qos_info_rel_left_{"reliable"},   qos_info_dur_left_{"volatile"};
  std::string qos_info_rel_right_{"reliable"},  qos_info_dur_right_{"volatile"};
  std::string qos_nitros_rel_left_{"best_effort"},  qos_nitros_dur_left_{"volatile"};
  std::string qos_nitros_rel_right_{"best_effort"}, qos_nitros_dur_right_{"volatile"};

  // Per-camera publish control (from cam_{left,right} params)
  bool        pub_raw_left_{true};           // cam_left.publish_image_raw
  bool        pub_raw_right_{true};          // cam_right.publish_image_raw
  std::string nitros_fmt_left_{"nv12"};     // cam_left.nitros_format
  std::string nitros_fmt_right_{"nv12"};    // cam_right.nitros_format

  // Static TF broadcaster (publishes extrinsics: relative_to → camera frame_id)
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

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

  // try_build_pipeline: attempts to set the pipeline to PLAYING.
  // nvmm  – request memory:NVMM caps from v4l2src and nvvidconv
  // out_fmt – "NV12" or "BGRx"; the nvvidconv output format
  bool try_build_pipeline(bool nvmm, const std::string & out_fmt = "BGRx");
  void build_pipeline();
  void capture_loop();
  void process_sample(GstSample * sample);

  // ── Publishers ────────────────────────────────────────────────────────────
  // Raw image publishers (sensor_msgs/Image, bgr8).
  // Created only when cam_{left,right}.publish_image_raw is true (default: true).
  // nullptr when disabled — callers must check pub_raw_left_/right_ flag before use.
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      pub_left_raw_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      pub_right_raw_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_left_info_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_right_info_;

  // NITROS publishers: GPU-resident zero-copy topic <prefix>/nitros_image_<format>.
  // Format is cam_{left,right}.nitros_format param (default: "nv12").
  // Created only when HAVE_NVBUF is compiled in.
#ifdef HAVE_NVBUF
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr pub_left_nitros_;
  rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosImage>::SharedPtr pub_right_nitros_;
  // Full NITROS topic names (= prefix + "/nitros_image_" + format)
  std::string topic_left_nitros_;
  std::string topic_right_nitros_;
#endif

  // ── Private helpers ───────────────────────────────────────────────────────
  /// Build a sensor_msgs::CameraInfo from the inline intrinsic params for the given side.
  sensor_msgs::msg::CameraInfo build_cam_info(const std::string & side, int w, int h);
  /// Broadcast a static TF transform (extrinsics) for one camera side.
  void broadcast_static_tf(const std::string & side, const std::string & child_frame);

  /// Effective published width/height for one eye (respects output_resolution param).
  int eff_out_w(bool left) const {
    int v = left ? out_w_left_ : out_w_right_;
    return (v > 0) ? v : eye_width();
  }
  int eff_out_h(bool left) const {
    int v = left ? out_h_left_ : out_h_right_;
    return (v > 0) ? v : combined_height_;
  }

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
  //   2. NvBufSurfTransformAsync ×2 → VIC NV12→NV12 crop per eye
  //      (pure geometric op, no colour conversion — VIC handles it natively)
  //   3. Wait for both VIC sync objects.
  //   4. gst_buffer_unmap — release NVMM src reference.
  //   5. nvbuf_{left,right}_->surfaceList[0].dataPtr → CUDA device pointer.
  //   6. NitrosImageBuilder::WithGpuData(cuda_ptr) → NitrosImage  [zero host copy]
  //   7. pub_{left,right}_->publish(nitros_img)  — GPU handle delivered zero-copy
  //      to co-located NITROS subscribers.
#ifdef HAVE_NVBUF
  // GXF shared context required by NitrosImageBuilder::Build() to allocate
  // GXF VideoBuffer entities.  Initialised in the constructor.
  nvidia::isaac_ros::nitros::NitrosContext nitros_ctx_;

  bool           use_nvbuf_{false};       // true once surfaces allocated successfully
  NvBufSurface * nvbuf_left_{nullptr};    // NV12 SURFACE_ARRAY VIC dst, left eye
  NvBufSurface * nvbuf_right_{nullptr};   // NV12 SURFACE_ARRAY VIC dst, right eye
  // Packed NV12 CUDA device buffers (cudaMalloc eye_w×h×3/2 bytes each).
  // After each VIC transform, cudaMemcpy2D copies Y+UV from the CPU-mapped
  // SURFACE_ARRAY into these, then NitrosImageBuilder::WithGpuData() wraps them.
  // On Jetson iGPU (unified LPDDR5), the cudaMemcpy2D is a coherence op only;
  // no physical data moves between GPU and a "separate" CPU.
  void *         cuda_nv12_left_{nullptr};
  void *         cuda_nv12_right_{nullptr};

  // Allocate NV12 CUDA-device NvBufSurfaces per eye; set use_nvbuf_ = true.
  void init_nvbuf_surfaces();
  // Load GXF cuda + multimedia extensions into the shared context so that
  // NitrosImageBuilder::Build() can create nvidia::gxf::VideoBuffer entities.
  // Must be called before the first Build() call (i.e. before the capture loop).
  void init_nitros_context();
  void cleanup_nvbuf_surfaces();

  // Returns true if the VIC path completed successfully; false causes
  // process_sample() to fall through to the CPU path.
  bool process_sample_nvbuf(GstBuffer * buf, const rclcpp::Time & stamp);
#endif  // HAVE_NVBUF
};

}  // namespace arducam_dual_camera


#endif  // ARDUCAM_DUAL_CAMERA__ARDUCAM_DUAL_CAM_NODE_HPP_