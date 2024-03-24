#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <memory>
#include "MSCK_Filter.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "MSCK_Filter");
  std::cout << "===node MSCK_Filter starts===" << std::endl;

  MSCKalman::MSCK_Filter msckf;
  msckf.init();
  //std::cout << "===node MSCK_Filter starts===" << std::endl;

  ros::spin();
  return 0;
}