#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <memory>
#include <cmath>
#include <math_utils.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace MSCKalman {

class MSCK_Filter {
 public:

  struct EulerAngles {
   double roll, pitch, yaw;
  };

  MSCK_Filter(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  MSCK_Filter()
      : MSCK_Filter(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~MSCK_Filter() {}
  //Prediction Terms
  //State Vector
  Eigen::VectorXd x; //OVERALL STATE VECTOR
  //IMU State Vectors
  Eigen::VectorXd x_imu; //IMU STATE VECTOR (Combination of Vectors Below)
  Eigen::Quaternion rotation_q; //Rotation Quaternion of IMU Frame wrt to Global Frame   
  Eigen::MatrixXd rotation_matrix; //Rotational Matrix of Quaternion
  Eigen::VectorXd gyr_bias; //Gyroscope Bias
  Eigen::VectorXd imu_vel; //Velocity of IMU
  Eigen::VectorXd acc_bias; //Accelerometer Bias
  Eigen::VectorXd imu_pos; //Position of IMU Frame wrt to Global Frame


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

  //Old Stuff
  //prediction terms
  Eigen::VectorXd x; //State Posterior Vector
  Eigen::MatrixXd P; //State Posterior Covariance
  Eigen::VectorXd u; //Control Input Vector
  Eigen::MatrixXd Q; //Control Input Covariance
  Eigen::MatrixXd F; //Motion Model Jacobian
  
  //correction terms
  Eigen::VectorXd xp; //State Estimate/Prior Vector
  Eigen::VectorXd yp; //State Estimate/Prior Vector - motion model
  Eigen::VectorXd xo; //Previous Estimate
  Eigen::VectorXd yo; //Previous Estimate Vector
  Eigen::MatrixXd Pp; //State Estimate/Prior Covariance
  Eigen::MatrixXd K; //Kalman Gain
  Eigen::VectorXd y; //Measurement Vector 
  Eigen::MatrixXd R; //Measurement Covariance
  Eigen::MatrixXd G; //Observation Model Jacobian
  
  //Jacobian Calculations
  Eigen::MatrixXd Qp; //Motion Model Jacobian Noise
  Eigen::MatrixXd Rp; //Observation Model Jacobian Noise
  Eigen::MatrixXd wp; //OBservation Model Derivative Noise

  //useful constants
  Eigen::MatrixXd I; //Identity Matrix 
  Eigen::MatrixXd x; 
  double t_0 = 0.0;
  double dt = 0.0;
  //Constants per measurements:
  double dti = 0.0; // IMU Time Step
  double dtc = 0.0; // Camera Time Step
  int N = 0;
  int Nmax =10;  
  
  //main functions
  void init();
  //void odom_Callback(const nav_msgs::Odometry &msg);
  void IMU_Callback(const sensor_msgs::ImuConstPtr& IMU_Msg);
  void propagate_imu(const Vector3d& acc_m, const Vector3d& gyr_m);
  Eigen::Matrix3d skew_symmetric(const Vector3d& vec);

  //Old Functions
  void Iterated_Extended_Kalman_Filter();
  //prediction functions
  void state_prediction();
  void covariance_prediction();
  //kalman gain function
  void kalman_gain();
  //correction functions
  void state_correction();
  void state_correction_I();
  void covariance_correction();
  void covariance_correction_I();
  //Jacobian Calculators
  void motion_model_Jacobian();
  void motion_noise_Jacobian();
  void observation_model_Jacobian();
  void observation_model_Jacobian_O();
  void observation_noise_Jacobian();
  void observation_noise_Jacobian_O();

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber infoSub_;
  ros::Publisher posePub_;
};

}  // namespace Kalman