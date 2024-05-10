#ifndef MSCKF_NODE_H
#define MSCKF_NODE_H

#include <ros/ros.h>
#include <nav_msgs/Image.h>
#include <sensor_msgs/Imu.h>

namespace MSCKalman {

class MSCKF_Node {
public:
    MSCKF_Node(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
    ~MSCKF_Node() {}

private:
    void IMU_Callback(const sensor_msgs::ImuConstPtr& IMU_Msg);
    void CAM_Callback(const nav_msgs::Image::ConstPtr& CAM_Msg);
    void Feature_Callback();
    void publish_odom();

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber imu_sub;
    ros::Subscriber cam_sub;
    ros::Publisher odom_pub;

    MSCKF_Filter msckf_filter; //Instance of Filter Functions
    
};

} // namespace MSCKalman

#endif // MSCKF_NODE_H