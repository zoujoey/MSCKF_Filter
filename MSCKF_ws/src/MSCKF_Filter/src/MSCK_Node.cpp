#include "MSCKF_Node.h"
#include "MSCKF_Filter.h"
namespace MSCKalman {

MSCKF_Node::MSCKF_Node(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {
    imu_sub = nh_.subscribe<sensor_msgs::Imu>("imu0", 1, &MSCKF_Node::IMU_Callback, this);
    ROS_INFO("Subscribe to IMU");
    cam_sub = nh_.subscribe<nav_msgs::Image>("cam0/image_raw", 1, &MSCKF_Node::CAM_Callback, this);
    ROS_INFO("Subscribe to CAM");
    msckf_filter.init(); // Call initialization function from MSCKF_Filter
}

void MSCKF_Node::IMU_Callback(const sensor_msgs::ImuConstPtr& IMU_Msg) {
    acc_m << IMU_Msg->linear_acceleration.x, IMU_Msg->linear_acceleration.y, IMU_Msg->linear_acceleration.z;
    gyr_m << IMU_Msg->angular_velocity.x, IMU_Msg->angular_velocity.y, IMU_Msg->angular_velocity.z;

    double imu_time = IMU_Msg->header.stamp.toSec();

    if(!is_gravity_init){
       imu_init_buffer.push_back(*IMU_Msg);
       if(imu_init_buffer.size() > 200){
           ROS_INFO("Initalization Starts");
           gravity_bias_initialization();
           is_gravity_init = true;
       }
    }
    else{
       ROS_INFO("Start propagation");
       propagate_imu(imu_dt, acc_m, gyr_m);
    }

    if(imu_first_data){
        imu_first_data = false;
        imu_last_time = imu_time;
        return;
    }
    publishOdom(); 
    imu_dt = imu_time - imu_last_time;
    imu_last_time = imu_time;
}

void MSCKF_Node::CAM_Callback(const nav_msgs::Image::ConstPtr& CAM_Msg) {
    // Implement your camera callback function here
    Visual_Odometry(CAM_Msg);

    if(is_gravity_init){
        ROS_INFO("Image Received");
        add_Image();
        state_augmentation();
        covariance_augmentation();
        N += 1;
        if(feature_disappeared){
        Eigen::Vector2D feature_estimate = feature_collapse(feature);
        Eigen::Vector2D
        }
    }
    publishOdom();
}

void MSCKF_Node::Feature_Callback() {
    // Implement your feature callback function here
}

void MSCKF_Filter::publishOdom() {
    // Implement your odom publishing function here
    imu_odom.header.frame_id = "world";
    imu_odom.pose.pose.position.x = imu_pos(0);
    imu_odom.pose.pose.position.y = imu_pos(1);
    imu_odom.pose.pose.position.z = imu_pos(2);

    imu_odom.pose.pose.orientation.x = rotation_q.x();
    imu_odom.pose.pose.orientation.y = rotation_q.y();
    imu_odom.pose.pose.orientation.z = rotation_q.z();
    imu_odom.pose.pose.orientation.w = rotation_q.w();

    odom_pub.publish(imu_odom);
}
} // namespace MSCKalman