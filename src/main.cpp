#include <rclcpp/rclcpp.hpp>
#include "arducam_dual_camera/arducam_dual_cam_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<arducam_dual_camera::ArducamDualCamNode>());
  rclcpp::shutdown();
  return 0;
}
