cmake_minimum_required(VERSION 3.8)
project(arducam_dual_camera)

# ── Compiler settings ─────────────────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall -Wextra
    # O3 + full AArch64 ISA (Cortex-A78AE on Orin Nano):
    #   simd    → NEON/SVE vector ops
    #   fp16    → native half-precision arithmetic
    #   dotprod → UDOT/SDOT for int8 convolutions
    -O3
    -march=armv8.2-a+simd+fp16+dotprod
    # Allow reassociation of floating-point ops for SIMD vectorisation.
    # Safe for image/video math; avoid if strict IEEE-754 is required elsewhere.
    -ffast-math
  )
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ROS 2 dependencies ────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# ── OpenCV (system, includes CUDA modules on Jetson) ─────────────────────────
find_package(OpenCV REQUIRED)

# ── Isaac ROS NITROS (zero-copy GPU image transport) ────────────────────
# NitrosImage publisher is only compiled when HAVE_NVBUF is set.
# find_package succeeds on any JetPack+Isaac ROS install; cmake does not error
# if the variable HAVE_NVBUF is ultimately false (the headers are not included).
find_package(isaac_ros_nitros REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)
find_package(ament_index_cpp REQUIRED)

# ── GStreamer (system) ────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0
)

# ── NVIDIA NvBufSurface / NvBufSurfTransform (DeepStream 7, Jetson Orin) ──────
# These libraries expose the VIC hardware API for zero-copy NVMM buffer access
# and hardware-accelerated crop + colour-space conversion.
# Header:  <nvbufsurface.h>   → surface allocation, map/unmap, syncForCpu
# Header:  <nvbufsurftransform.h> → async VIC crop+convert transforms
# Both are installed by the deepstream-7.x package on JetPack 6.
set(DS_ROOT "/opt/nvidia/deepstream/deepstream-7.1" CACHE PATH
    "DeepStream SDK root; set to '' to disable NvBufSurface path")

if(EXISTS "${DS_ROOT}/sources/includes/nvbufsurface.h" AND
   EXISTS "${DS_ROOT}/lib/libnvbufsurface.so"          AND
   EXISTS "${DS_ROOT}/lib/libnvbufsurftransform.so")
  set(HAVE_NVBUF TRUE)
  message(STATUS "NvBufSurface found at ${DS_ROOT} — VIC crop+convert path enabled")
else()
  set(HAVE_NVBUF FALSE)
  message(STATUS "NvBufSurface NOT found — CPU colour-conversion fallback will be used")
endif()

set(ROS_DEPS
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  cv_bridge
  tf2
  tf2_ros
  isaac_ros_nitros
  isaac_ros_nitros_image_type
  ament_index_cpp
)

# ── Composable component shared library ───────────────────────────────────────
# Building as a shared library enables:
#   • Loading into a composable container → intra-process zero-copy to
#     DnnImageEncoderNode, VIO, and any other co-located subscribers.
#   • rclcpp_components_register_node() auto-generates a standalone wrapper
#     executable (arducam_dual_cam_node) for use outside a container.
add_library(arducam_dual_camera_component SHARED
  src/arducam_dual_cam_node.cpp
)

target_include_directories(arducam_dual_camera_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GST_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(arducam_dual_camera_component
  ${GST_LIBRARIES}
  ${OpenCV_LIBS}
)

# ── Optional NvBufSurface / NvBufSurfTransform + Isaac ROS NITROS ─────────────
# When HAVE_NVBUF is true (Jetson + DeepStream) we also bring in the NITROS image
# type so that VIC-output NvBufSurfaces can be published zero-copy as NitrosImage,
# which downstream isaac_ros nodes consume without any host DMA or memcpy.
if(HAVE_NVBUF)
  # isaac_ros_nitros and isaac_ros_nitros_image_type are already found
  # unconditionally above; no need to call find_package again here.
  list(APPEND ROS_DEPS
    isaac_ros_nitros
    isaac_ros_nitros_image_type
  )

  target_include_directories(arducam_dual_camera_component PRIVATE
    ${DS_ROOT}/sources/includes
  )
  target_link_libraries(arducam_dual_camera_component
    ${DS_ROOT}/lib/libnvbufsurface.so
    ${DS_ROOT}/lib/libnvbufsurftransform.so
    # isaac_ros_nitros_image_type is linked transitively via ament_target_dependencies
  )
  target_compile_definitions(arducam_dual_camera_component PRIVATE HAVE_NVBUF=1)
endif()

ament_target_dependencies(arducam_dual_camera_component ${ROS_DEPS})

# Register the node as a component and generate the standalone executable
rclcpp_components_register_node(arducam_dual_camera_component
  PLUGIN "arducam_dual_camera::ArducamDualCamNode"
  EXECUTABLE arducam_dual_cam_node
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS arducam_dual_camera_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/list_camera_modes.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
