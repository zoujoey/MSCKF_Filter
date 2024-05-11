#ifndef MSCKF_NODE_H
#define MSCKF_NODE_H

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cv_bridge/cv_bridge.h>

#include "MSCKF_Filter/MSCK_Filter.h"
#include "MSCKF_Filter/ImageFeatures.h"
#include "MSCKF_Filter/Feature.h"
#include "MSCKF_Filter/math_utils.h"

namespace MSCKalman {

class MSCKF_Node {
public:
    MSCKF_Node(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
    ~MSCKF_Node() {}

private:
    //Callback Functions
    void IMU_Callback(const sensor_msgs::ImuConstPtr &IMU_Msg);
    void IMAGE_Callback(const sensor_msgs::ImageConstPtr &CAM_Msg);
    void Feature_Callback(const MSCKalman::ImageFeaturesConstPtr &IMG_Msg);
    
    //Odometry Functions
    void publish_odom();

    //Subscribers
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber imu_sub;
    image_transport::Subscriber image_sub;
    ros::Subscriber features_sub;

    //Publishers
    ros::Publisher odom_pub;

    MSCKF_Filter msckf_filter; //Instance of Filter Functions

};

} // namespace MSCKalman

#endif // MSCKF_NODE_H