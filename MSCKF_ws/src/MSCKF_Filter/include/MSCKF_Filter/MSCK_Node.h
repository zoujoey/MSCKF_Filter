#ifndef MSCKF_NODE_H
#define MSCKF_NODE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <image_transport/image_transport.h>
#include "MSCKF_Filter/ImageFeatures.h"
#include "MSCKF_Filter/MSCK_EKF.h"

namespace MSCKalman {

class MSCKF_Node {
public:
    MSCKF_Node(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
    MSCKF_Node()
      : MSCKF_Node(ros::NodeHandle(), ros::NodeHandle("~")) {}
    ~MSCKF_Node() {}

private:
    //Callback Functions
    void imu_callback(const sensor_msgs::ImuConstPtr &msg);
    void image_callback(const sensor_msgs::ImageConstPtr &msg);
    void feature_callback(const MSCKF_Filter::ImageFeaturesConstPtr &msg);
    
    //Odometry Functions
    void publish_odom();

    //MSCKF_Filter::ImageFeatures test;

    //Subscribers
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber imu_sub;
    image_transport::Subscriber image_sub;
    ros::Subscriber features_sub;

    //Publishers
    ros::Publisher odom_pub;

    MSCKF_EKF filter; //Instance of Filter Functions

};

} // namespace MSCKalman

#endif // MSCKF_NODE_H