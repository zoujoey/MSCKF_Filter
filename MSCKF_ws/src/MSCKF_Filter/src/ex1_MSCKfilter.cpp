#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <random>
#include <cmath>

class RandomPoseGenerator {
public:
    RandomPoseGenerator(double stddev) : distribution(0.0, stddev) {}

    double addGaussianNoise(double value) {
        return value + distribution(generator);
    }

private:
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
};

// Function to generate random motion parameters
void randomMotionParameters(double& r, double& v0) {
    r = 5 + (rand() % 3);
    v0 = (1 + (rand() % 4)) / 2.50;
}

// Function to calculate motion equations
void motionEquations(double t, double r, double v0, double& x, double& y, double& vx, double& vy) {
    // Calculate position equations
    x = r * cos(v0 * t);
    y = r * sin(v0 * t);

    // Calculate velocity equations with correct signs
    vx = -r * v0 * sin(v0 * t);
    vy = r * v0 * cos(v0 * t);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ex1_EKfilter");
    ros::NodeHandle n;
    ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("true_pose_topic", 1000);
    ros::Publisher pos_pub = n.advertise<nav_msgs::Odometry>("noisy_pose_topic", 1000);  // New topic for position data
    ros::Rate loop_rate(10);

    // Generate random motion parameters once
    double r, v0;
    randomMotionParameters(r, v0);

    // Specify standard deviations for Gaussian noise
    double pose_noise_stddev = 1;  // Adjust as needed

    // Create a RandomPoseGenerator instance for adding noise
    RandomPoseGenerator poseNoiseGenerator(pose_noise_stddev);

    int count = 0;
    double t = 0.0;
    double phi = 3.141592/2;
    // Record the start time
    ros::Time start_time = ros::Time::now();

    while (ros::ok()) {
        nav_msgs::Odometry msg;

        // Calculate elapsed time
        ros::Time current_time = ros::Time::now();
        ros::Duration elapsed_time = current_time - start_time;
        t = elapsed_time.toSec();
        ros::Time elapsed_time_2 = ros::Time(t);

        // Calculate position, velocity, and acceleration at time t
        double x, y, vx, vy;
        motionEquations(t, r, v0, x, y, vx, vy);

        // Publish actual position to pos topic
        nav_msgs::Odometry pos_msg;
        pos_msg.header.stamp = elapsed_time_2;  // Set the timestamp to the same as the main message
        pos_msg.header.frame_id = "World";
        pos_msg.pose.pose.position.x = x;
        pos_msg.pose.pose.position.y = y;
        pos_msg.pose.pose.position.z = 0;
        pos_msg.twist.twist.linear.x = r*v0;
        pos_msg.twist.twist.linear.y = v0;
        pos_msg.twist.twist.linear.z = 0;
        odom_pub.publish(pos_msg);
        std::cout << "actual position of the drone: " << x << "," << y << "," << phi << "," << r << "," << t << std::endl;

        // Add Gaussian noise to pose data
        phi = 3.141592/2;
        vx = r*v0;
        double nr = poseNoiseGenerator.addGaussianNoise(r);
        phi = poseNoiseGenerator.addGaussianNoise(phi);

        // Fill in the Odometry message
        msg.header.stamp = elapsed_time_2;
        msg.header.frame_id = "World";
        msg.pose.pose.position.x = nr;
        msg.pose.pose.position.y = phi;
        msg.twist.twist.linear.x = vx;
        msg.twist.twist.linear.y = v0;
        tf2::Quaternion quaternion;

        pos_pub.publish(msg);

        std::cout << "noisy position of the drone: " << x << "," << y << "," << phi << "," << r << "," << t << std::endl;

        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }

    return 0;
}
