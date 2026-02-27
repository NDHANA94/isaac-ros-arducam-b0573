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

  * @file main.cpp
  * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
  * @date: 27.02.2026
  * @brief Main entry point for the Arducam B0573 (GMSL2-to-CSI2) dual camera ROS 2 node.
*/


#include <rclcpp/rclcpp.hpp>
#include "isaac_ros_arducam_b0573/arducam_b0573_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<isaac_ros_arducam_b0573::ArducamB0573Node>());
  rclcpp::shutdown();
  return 0;
}
