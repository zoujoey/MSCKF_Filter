#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <memory>
#include "MSCKF_Filter/MSCK_Node.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "MSCK_Node");
  std::cout << "===node MSCK_Node starts===" << std::endl;

  MSCKalman::MSCKF_Node msck_node;
  
  ros::spin();
  return 0;
}
