#include <rclcpp/rclcpp.hpp>
#include "isaac_ros_arducam_b0573/arducam_b0573_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<isaac_ros_arducam_b0573::ArducamB0573Node>());
  rclcpp::shutdown();
  return 0;
}
