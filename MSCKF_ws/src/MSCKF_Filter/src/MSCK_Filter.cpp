#include "MSCK_Filter.h"


namespace MSCKalman {

MSCK_Filter::MSCK_Filter(const ros::NodeHandle &nh,
                                 const ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh) {
  infoSubIMU_ = nh_.subscribe<sensor_msgs::Imu>("noisy_pose_topic_IMU", 1, &MSCK_Filter::IMU_Callback, this);
  //infoSubCAM_ = nh_.subscribe("noisy_pose_topic_CAM", 1, &MSCK_Filter::odom_Callback, this);
  posePub_ = nh_.advertise<nav_msgs::Odometry>("est_pose_topic", 1000);
  double t_0 = 0;
  double dt = 0;
}
void MSCK_Filter::init(){
  // Initialize Prior Estimates
  x = Eigen::VectorXd::Zero(16+7*N); // State Posterior Vector
  x_imu = Eigen::VectorXd::Zero(16); //Imu State Vector
  rotation_q = Eigen::Quaternion::Identity();
  rotation_matrix.resize(3,3);//Rotational Matrix of Quaternion
  //Rotate when needed during IMU propagation
  //rotation_matrix = rotation_q.normalized().toRotationMatrix();
  gyr_bias = Eigen::VectorXd::Zero(3);//Gyrscope Bias
  imu_vel = Eigen::VectorXd::Zero(3);//Velocity Vector
  acc_bias = Eigen::VectorXd::Zero(3);//Accelerometer Bias
  imu_pos = Eigen::VectorXd::Zero(3);//Position Vector
  cam_state = Eigen::VectorXd::Zero(7*N);//Camera State Vector
  cam_q = Eigen::VectorXd::Zero(7*N);//Camera State Rotation Recent
  cam_pos = Eigen::VectorXd::Zero(7*N);//Camera State Position Recent


  
  F.resize(15,15) // Error State Jacobian
  F = << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //1
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //2
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //3
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //4
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //5
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //6
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //7
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //8
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //9
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //10
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //11
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //12
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //13
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //14
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; //15

  G.resize(15,12) //Error State Noise Jacobian 
  G = << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //1
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //2
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //3
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //4
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //5
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //6
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //7
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //8
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //9
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //10
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //11
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //12
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //13
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //14
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; //15

  P.resize(15+6*N, 15+6*N) //TOTAL COVARIANCE MATRIX
  P = Eigen::MatrixXd::Zero(15+6*N, 15+6*N);
  Pii.resize(15, 15) //Covariance Matrix of IMU
  Pii = Eigen::MatrixXd::Zero(15, 15);
  Pcc.resize(6*N, 6*N) //Covariance Matrix of Camera
  Pcc = Eigen::MatrixXd::Zero(6*N, 6*N);
  Pic.resize(15, 6*N) //Correlation Matrix between IMU and Camera
  Pic = Eigen::MatrixXd::Zero(15, 6*N);

  acc_m = Eigen::VectorXd::Zero(3); //Acclerometer vector
  gyr_m = Eigen::VectorXd::Zero(3); //Gyrscope reading
  acc_skew.resize(3,3); //skew matrix accelerometer
  acc_skew = Eigen::MatrixXd::Zero(3,3);
  gyr_skew.resize(3,3); //skew matrix gyroscope
  gyr_skew = Eigen::MatrixXd::Zero(3,3);
  imu_noise = Eigen::VectorXd::Zero(3+3+3+3);//IMU measurement Noise
  Q_imu.resize(12,12);
  Q_imu = Eigen::MatrixXd::Zero(12,12); //Covariance Matrix of IMU Noise
  phi.resize(15,15);
  //phi = Eigen::MatrixXd::Identity(); 
}

void MSCK_Filter::IMU_Callback(const sensor_msgs::ImuConstPtr& IMU_Msg){
    //IMU acceleration and angular velocity measurements
    acc_m << IMU_Msg->linear_acceleration.x, IMU_Msg->linear_acceleration.y, IMU_Msg->linear_acceleration.z;
    gyr_m << IMU_Msg->angular_velocity.x, IMU_Msg->angular_velocity.y, IMU_Msg->angular_velocity.z;

    propagate_imu(acc_m, gyr_m);
}

void MSCK_Filter::propagate_imu(const Eigen::Vector3d& acc_m, const Eigen::Vector3d& gyr_m){
    Eigen::Vector3d acc = acc_m - acc_bias;
    Eigen::Vector3d gyr = gyr_m - gyr_bias;

    Eigen::Matrix3d rotation_matrix = rotation_q.normalized().toRotationMatrix();

    //Transition (F) and Noise (G) Matrix
    //We assume planet rotation is negligible (w_g = 0)
    F.block<3,3>(0,0) = -skew_symmetric(gyr);
    F.block<3,3>(0,3) = -Eigen::Matrix3d::Identity();
    F.block<3,3>(6,0) = -rotation_matrix.transpose()*skew_symmetric(acc);
    F.block<3,3>(6,9) = -rotation_matrix.transpose();
    F.block<3,3>(12,6) = Eigen::Matrix3d::Identity();

    G.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
    G.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
    G.block<3,3>(6,6) = -rotation_matrix.transpose();
    G.block<3,3>(9,9) = Eigen::Matrix3d::Identity();

    //Phi (State Transition Matrix)
    //Approximate Matrix Exponential Using Taylor Series
    //3rd Order For Now
    //Eigen::MatrixXd F = F * dt;
    Eigen::MatrixXd F_2 = F*F;
    Eigen::MatrixXd F_3 = F*F*F;
    phi = Eigen::MatrixXd::Identity() + F*dt + (1/2)*F_2*dt*dt + (1/6)*F_3*dt*dt*dt;

    //TODO:
    //State propagation with runge kutta
    //Other Stuff
}

//Maybe put in separate util file?
Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& vec){
    Eigen::Matrix3d vec_skew;
    vec_skew << 0, -vec(2), vec(1),
                vec(2), 0, -vec(0),
                -vec(1), vec(0), 0;
    return vec_skew;
}

// void MSCK_Filter::odom_Callback(const nav_msgs::Odometry &msg) {
//     // Assuming that the position information is in the pose field of Odometry message
//     y   << msg.pose.pose.position.x,
//            msg.pose.pose.position.y;
//     u   << msg.twist.twist.linear.x,
//            msg.twist.twist.linear.y;
//     if (t_0 == 0){
//       t_0 = msg.header.stamp.toSec(); 
//     x   << y(0),0, -3.141592/2;
//     }
//     else{
//       dt = msg.header.stamp.toSec()-t_0;
//       Iterated_Extended_Kalman_Filter();
//       t_0 = msg.header.stamp.toSec();
//     }
//     nav_msgs::Odometry fmsg;
//     fmsg.header.stamp = msg.header.stamp;
//     fmsg.header.frame_id = "World";
//     tf2::Quaternion quat;
//     quat.setRPY(0, 0, x(2));
//     fmsg.pose.pose.position.x = x(0);
//     fmsg.pose.pose.position.y = x(1);
//     fmsg.pose.pose.orientation = tf2::toMsg(quat);
//     posePub_.publish(fmsg);
//     std::cout << "filter-based position of the drone: " << x(0) << "," << x(1) << "," << x(2) << "," << t_0 << std::endl;
// }

void MSCK_Filter::Iterated_Extended_Kalman_Filter(){
    //Jacobian Calculations
    motion_model_Jacobian();
    motion_noise_Jacobian();
    observation_model_Jacobian();
    observation_noise_Jacobian();
    //Prediction Step
    state_prediction();
    covariance_prediction();
    xo = xp;
    int i = 0;
    while (i<20){
    xo = x;   
    observation_model_Jacobian_O();
    observation_noise_Jacobian_O();
    //Kalman Gain
    kalman_gain();
    //Correction Step
    covariance_correction_I();
    state_correction_I();
    i++;
    }
    }
void MSCK_Filter::state_prediction(){
       xp   << (x(0) + cos(x(2))*u(0)*dt),
               (x(1) + sin(x(2))*u(0)*dt),
               (x(2) + u(1)*dt);
       xp(2) = std::fmod(xp(2) + M_PI, 2.0 * M_PI) - M_PI;
}
void MSCK_Filter::covariance_prediction(){
       Pp = F * P * F.transpose() + Qp;
}
void MSCK_Filter::kalman_gain(){
       K = Pp * G.transpose() * (G * Pp * G.transpose() + Rp).inverse();
}
void MSCK_Filter::state_correction(){
       yp   <<(sqrt((xp(0)*xp(0))+(xp(1))*(xp(1)))), 
              (std::atan2(-xp(1),-xp(0)) - xp(2)); 
           // Ensure yp(1) is in the range [-pi, pi]
       if (yp(1) < -M_PI) {
           yp(1) += 2 * M_PI;
       } else if (yp(1) > M_PI) {
           yp(1) -= 2 * M_PI;
       }

       std::cout << "yp " << yp(0) <<" , " << yp(1)<< std::endl;
       std::cout << "y " << y(0) <<" , " << y(1)<< std::endl;
       x = xp + K * (y - yp);
}
void MSCK_Filter::state_correction_I(){
       yo   <<(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), 
              (std::atan2(-xo(1),-xo(0)) - xo(2)); 
           // Ensure yp(1) is in the range [-pi, pi]
       if (yo(1) < -M_PI) {
           yo(1) += 2 * M_PI;
       } else if (yo(1) > M_PI) {
           yo(1) -= 2 * M_PI;
       }

       std::cout << "yp " << yo(0) <<" , " << yo(1)<< std::endl;
       std::cout << "y " << y(0) <<" , " << y(1)<< std::endl;
       x = xp + K * (y - yo - G*(xp - xo));
}
void MSCK_Filter::covariance_correction(){
       P = (I - K * G) * Pp;
}
void MSCK_Filter::covariance_correction_I(){
       P = (I - K * G) * Pp;
}
void MSCK_Filter::motion_model_Jacobian(){
   F   << 1, 0, (-sin(x(2))*u(0)*dt),
          0, 1, (cos(x(2))*u(0)*dt),
          0, 0, 1;
}
void MSCK_Filter::motion_noise_Jacobian(){
   wp   <<  cos(x(2)), 0,
            sin(x(2)), 0,
            0,         1;
   Qp = dt*dt * wp * Q * wp.transpose();
}
void MSCK_Filter::observation_model_Jacobian(){
   G   << (x(0))/(sqrt((x(0)*x(0))+(x(1))*(x(1)))), (x(1))/(sqrt((x(0)*x(0))+(x(1))*(x(1)))), 0,
          (-x(1))/(sqrt((x(0)*x(0))+(x(1))*(x(1)))), (x(0))/(sqrt((x(0)*x(0))+(x(1))*(x(1)))), -1;
}
void MSCK_Filter::observation_model_Jacobian_O(){
   G   << (xo(0))/(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), (xo(1))/(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), 0,
          (-xo(1))/(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), (xo(0))/(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), -1;
}
void MSCK_Filter::observation_noise_Jacobian(){
   Rp = R;
}
void MSCK_Filter::observation_noise_Jacobian_O(){
   Rp = R;
}
}
// namespace Kalman
