#include <nodelet/loader.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sstream>
#include <string>
//#include "MSCKF_Filter/tracking_nodelet.h"
#include "MSCKF_Filter/tracking_descriptor.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "feature_tracking");
  std::cout << "===node feature_tracking starts===" << std::endl;

  //MSCKalman::FeatureTracker tracker;
  MSCKalman::TrackDescriptor tracker;
  
  ros::spin();
  return 0;
}