cmake_minimum_required(VERSION 3.8)
project(arducam_dual_camera)

# ── Compiler settings ─────────────────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O2)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── ROS 2 dependencies ────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_info_manager REQUIRED)

# ── OpenCV (system, includes CUDA modules on Jetson) ─────────────────────────
find_package(OpenCV REQUIRED)

# ── GStreamer (system) ────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0
)

set(ROS_DEPS
  rclcpp
  sensor_msgs
  cv_bridge
  camera_info_manager
)

# ── Main node executable ──────────────────────────────────────────────────────
add_executable(arducam_dual_cam_node
  src/arducam_dual_cam_node.cpp
)

target_include_directories(arducam_dual_cam_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${GST_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(arducam_dual_cam_node
  ${GST_LIBRARIES}
  ${OpenCV_LIBS}
)

ament_target_dependencies(arducam_dual_cam_node ${ROS_DEPS})

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS arducam_dual_cam_node
  DESTINATION lib/${PROJECT_NAME}
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
