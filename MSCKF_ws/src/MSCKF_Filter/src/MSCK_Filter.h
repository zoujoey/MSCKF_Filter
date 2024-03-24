#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <memory>
#include <vector>
#include <cmath>
//#include <math_utils.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

namespace MSCKalman {

class MSCK_Filter {
 public:
  MSCK_Filter(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  MSCK_Filter()
      : MSCK_Filter(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~MSCK_Filter() {}


  //main functions
  void init();
  void publishOdom();
  void IMU_Callback(const sensor_msgs::ImuConstPtr& IMU_Msg);
  void propagate_imu(double dt, const Eigen::Vector3d& acc_m, const Eigen::Vector3d& gyr_m);
  void imu_state_estimate(const double& dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc);
  Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& vec);
  void gravity_bias_initialization();

  //Prediction Terms
  //State Vector
  Eigen::VectorXd x; //OVERALL STATE VECTOR
  //IMU State Vectors
  Eigen::VectorXd x_imu; //IMU STATE VECTOR (Combination of Vectors Below)
  Eigen::Quaterniond rotation_q; //Rotation Quaternion of IMU Frame wrt to Global Frame   
  Eigen::MatrixXd rotation_matrix; //Rotational Matrix of Quaternion
  Eigen::VectorXd gyr_bias; //Gyroscope Bias
  Eigen::VectorXd imu_vel; //Velocity of IMU
  Eigen::VectorXd acc_bias; //Accelerometer Bias
  Eigen::VectorXd imu_pos; //Position of IMU Frame wrt to Global Frame
  std::vector<sensor_msgs::Imu> imu_init_buffer;

  //Gravity 
  Eigen::Vector3d gravity;
  bool is_gravity_init;

  //Camera Pose State Vectors
  Eigen::VectorXd cam_state; //CAMERA STATE VECTOR (Stores all Camera Poses Needed (up to N-max))
  Eigen::VectorXd cam_q; //Most Recent Quaternion of Camera wrt to Global Frame
  Eigen::VectorXd cam_pos; //Most Recent Position of Camera wrt to Global Frame

  //State Covariance
  Eigen::MatrixXd F; //Error State Jacobian
  Eigen::MatrixXd G; //Error State Noise Jacobian 
  Eigen::MatrixXd P; //TOTAL COVARIANCE MATRIX
  Eigen::MatrixXd Pii; //Covariance Matrix of IMU
  Eigen::MatrixXd Pic; //Correlation Matrix between IMU and Camera
  Eigen::MatrixXd Pcc; //Covariance Matrix of Camera
  
  //Measurement Model
  Eigen::VectorXd acc_m; //accelerometer reading
  Eigen::VectorXd gyr_m; //gyroscope reading
  Eigen::MatrixXd acc_skew; //skew matrix acclerometer
  Eigen::MatrixXd gyr_skew; //skew matrix gyroscope
  Eigen::VectorXd imu_noise; //measurement noise of IMU
  Eigen::MatrixXd Q_imu; //Covariance Matrix of IMU Noise
  Eigen::MatrixXd phi; //State Transition Matrix

  //useful constants
  Eigen::MatrixXd I; //Identity Matrix 
  double imu_dt; //IMU Time Step
  double cam_dt; //Camera Time Step
  double imu_current_time;
  double imu_last_time;
  bool imu_first_data;
  int N; //num camera poses
  int N_max;

  //Odom Publisher
  nav_msgs::Odometry imu_odom;

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber imu_sub;
  ros::Publisher odom_pub;
};

}  // namespace Kalman