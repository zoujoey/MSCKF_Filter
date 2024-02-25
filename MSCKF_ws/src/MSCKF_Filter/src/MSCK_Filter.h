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

  struct Quaternion {
   double w, x, y, z;
  };

  struct EulerAngles {
   double roll, pitch, yaw;
  };

  MSCK_Filter(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  MSCK_Filter()
      : MSCK_Filter(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~MSCK_Filter() {}

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
  double t_0 = 0.0;
  double dt = 0.0;

  //main functions
  void init();
  void odom_Callback(const nav_msgs::Odometry &msg);
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