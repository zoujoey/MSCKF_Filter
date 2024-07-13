#include "MSCKF_Filter/MSCK_Node.h"
#include <chrono>

namespace MSCKalman {

MSCKF_Node::MSCKF_Node(const ros::NodeHandle& nh, const ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh) {
    imu_sub = nh_.subscribe<sensor_msgs::Imu>("imu0", 1, &MSCKF_Node::imu_callback, this);
    ROS_INFO("Subscribe to IMU");

    features_sub = nh_.subscribe("/features", 1, &MSCKF_Node::image_feature_callback, this);

    odom_pub = nh_.advertise<nav_msgs::Odometry>("odom", 10);

    std::cout << "Initialize Call" << std::endl;
    filter.init(); // Call initialization function from MSCKF_Filter
}

void MSCKF_Node::imu_callback(const sensor_msgs::ImuConstPtr& msg) {
    ROS_INFO("IMU Callback");
    filter.acc_m << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
    filter.gyr_m << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;

    double imu_time = msg->header.stamp.toSec();

    if(!filter.is_gravity_init){
       filter.imu_init_buffer.push_back(*msg);
       if(filter.imu_init_buffer.size() > 200){
           //ROS_INFO("Initalization Starts");
           filter.gravity_bias_initialization();
           filter.is_gravity_init = true;
       }
    }
    else{
       //ROS_INFO("Start propagation");
       filter.propagate_imu(filter.imu_dt, filter.acc_m, filter.gyr_m);
    }

    if(filter.imu_first_data){
        filter.imu_first_data = false;
        filter.imu_last_time = imu_time;
        return;
    }   
    publish_odom(); 
    filter.imu_dt = imu_time - filter.imu_last_time;
    filter.imu_last_time = imu_time;
}

void MSCKF_Node::image_feature_callback(const MSCKF_Filter::ImageFeaturesConstPtr &msg) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Implement your feature callback function here
    if (filter.is_gravity_init) {
        // ROS_INFO("Image Received");
        filter.add_camera_frame(msg->image_seq);
        auto features = FeatureList{};
        for (const auto &f : msg->features) {
            features.push_back(ImageFeature{f.id, {f.position.x, f.position.y}, f.lifetime});
            // std::cout << "Num Features: " << features.size() << std::endl;
            if (f.lifetime >= 4) {
            //    std::cout << "Feature ID: " << f.id << " Lifetime: " << f.lifetime << " frames" << std::endl;
            }
        }
        filter.add_features(msg->image_seq, features);
    }
    std::cout << "Feature Callback: " << msg->image_seq << std::endl;
    publish_odom();

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Callback runtime: " << duration << " microseconds" << std::endl;
}


void MSCKF_Node::publish_odom() {
    // Implement your odom publishing function here
    ROS_INFO("Publish Odom");
    nav_msgs::Odometry imu_odom;
    imu_odom.header.frame_id = "world";
    imu_odom.pose.pose.position.x = filter.imu_pos(0);
    imu_odom.pose.pose.position.y = filter.imu_pos(1);
    imu_odom.pose.pose.position.z = filter.imu_pos(2);

    imu_odom.pose.pose.orientation.x = filter.rotation_q.x();
    imu_odom.pose.pose.orientation.y = filter.rotation_q.y();
    imu_odom.pose.pose.orientation.z = filter.rotation_q.z();
    imu_odom.pose.pose.orientation.w = filter.rotation_q.w();

    odom_pub.publish(imu_odom);
    
}
} // namespace MSCKalman