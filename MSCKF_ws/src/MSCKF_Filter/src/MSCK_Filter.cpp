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
  rotation_q = Eigen::Quaterniond::Identity();
  rotation_matrix.resize(3,3);//Rotational Matrix of Quaternion
  //Rotate when needed during IMU propagation
  //rotation_matrix = rotation_q.normalize().toRotationMatrix();
  gyr_bias = Eigen::VectorXd::Zero(3);//Gyrscope Bias
  imu_vel = Eigen::VectorXd::Zero(3);//Velocity Vector
  acc_bias = Eigen::VectorXd::Zero(3);//Accelerometer Bias
  imu_pos = Eigen::VectorXd::Zero(3);//Position Vector
  cam_state = Eigen::VectorXd::Zero(7*N);//Camera State Vector
  cam_q = Eigen::VectorXd::Zero(7*N);//Camera State Rotation Recent
  cam_pos = Eigen::VectorXd::Zero(7*N);//Camera State Position Recent
  N = 0;

  gravity = Eigen::Vector3d::Zero(); //Initialize gravity vector, set value later
  is_gravity_init = false;

  F.resize(15,15); // Error State Jacobian
  F << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //1
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

  G.resize(15,12); //Error State Noise Jacobian 
  G << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //1
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

  P.resize(15+6*N, 15+6*N); //TOTAL COVARIANCE MATRIX
  P = Eigen::MatrixXd::Zero(15+6*N, 15+6*N);
  Pii.resize(15, 15);//Covariance Matrix of IMU
  Pii = Eigen::MatrixXd::Zero(15, 15);
  Pcc.resize(6*N, 6*N); //Covariance Matrix of Camera
  Pcc = Eigen::MatrixXd::Zero(6*N, 6*N);
  Pic.resize(15, 6*N);//Correlation Matrix between IMU and Camera
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

    if(!is_gravity_init){
       imu_init_buffer.push_back(*IMU_Msg);
       if(imu_init_buffer.size() < 200){
           return;
       }
       else{
           gravity_bias_initialization();
           is_gravity_init = true;
       }
    }
    else{
       propagate_imu(acc_m, gyr_m);
    }
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

    Eigen::MatrixXd I_15 = Eigen::MatrixXd::Identity(15,15);

    //Phi (State Transition Matrix)
    //Approximate Matrix Exponential Using Taylor Series
    //3rd Order For Now
    Eigen::MatrixXd F = F * dt;
    Eigen::MatrixXd F_2 = F*F;
    Eigen::MatrixXd F_3 = F*F*F;
    phi = I_15 + F*dt + (1/2)*F_2*dt*dt + (1/6)*F_3*dt*dt*dt;
    //Discrete Time Noise Covariance Matrix Qk
    Eigen::MatrixXd Qk = phi*G*Q_imu*G*phi.transpose()*dt;
    //Covariance Propagation for IMU
    Eigen::MatrixXd Pii_1 = phi*Pii*phi.transpose()+Qk;
    //Total Covariance Propagation
    Eigen::MatrixXd P_1;

    int temp = 6*N;

    P.resize(15+6*N, 15+6*N);
    P_1.block<15,15>(0,0) = Pii_1;
    //P_1.block<temp,15>(15,0) = (P.block<temp,15>(15,0))*phi.transpose();
    P_1.block(15, 0, temp, 15) = (P.block(15,0, temp, 15))*phi.transpose();
    //P_1.block<15,temp>(0,15) = phi*(P.block<15,temp>(0,15));
    P_1.block(15, 0, 15, temp) = phi*(P.block(0,15,15,temp));
    //P_1.block<temp,temp(15,15) = P.block<temp,temp>(15,15);
    P_1.block(15,15, temp, temp) = P.block(15, 15, temp, temp);
    P_1 = (P_1+P_1.transpose())*0.5;

    imu_state_estimate(dt, gyr, acc);    
}

void MSCK_Filter::gravity_bias_initialization(){
    Eigen::Vector3d sum_gyr = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_gravity = Eigen::Vector3d::Zero();

    for(const auto& IMU_Msg : imu_init_buffer){
       Eigen::Vector3d temp_acc_m = Eigen::Vector3d::Zero();
       Eigen::Vector3d temp_gyr_m = Eigen::Vector3d::Zero();

       tf::vectorMsgToEigen(IMU_Msg.linear_acceleration, temp_acc_m);
       tf::vectorMsgToEigen(IMU_Msg.angular_velocity, temp_gyr_m);

       sum_acc += temp_acc_m;
       sum_gyr += temp_gyr_m;
    }

    gyr_bias = sum_gyr / imu_init_buffer.size();
    imu_gravity = acc_m / imu_init_buffer.size();
    

    gravity(3) = -(imu_gravity.norm());

    //may have to adjust rotation initalization
    Eigen::Vector3d v = gravity.cross(-imu_gravity);
    double s = v.norm();
    double c = gravity.dot(-imu_gravity);
    Eigen::Matrix3d rotation_c = Eigen::Matrix3d::Identity() + skew_symmetric(v) + (skew_symmetric(v))*(skew_symmetric(v))*((1 - c) / s*s);
    rotation_q = Eigen::Quaterniond(rotation_c);

}

void MSCK_Filter::imu_state_estimate(const double& dt, const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc){
    //IMU Quaternion Integration (0th Order)
    double gyr_norm = gyr.norm(); //get magnitude of gyro
    Eigen::Matrix4d omega = Eigen::Matrix4d::Zero();
    omega.block<3,3>(0,0) = -skew_symmetric(gyr);
    omega.block<3,1>(0,3) = gyr;
    omega.block<1,3>(3,0) = -gyr;

    //Integrate quaternion differently depending on size of gyr
    //Very small gyr will cause numerically instability
    Eigen::Quaterniond q_t_dt;
    Eigen::Quaterniond q_t_dt2;

    Eigen::Vector4d q_t_vec;
    Eigen::Vector4d q2_t_vec;
    Eigen::Vector4d q_vec;
    q_vec << rotation_q.w(), rotation_q.x(), rotation_q.y(), rotation_q.z();
    //Eigen::Vector4d q2_vec;

    if(gyr_norm > 1e-5) { //tune this parameter, current parameter taken from existing implementation
       //Use standard zero-th order integrator
       q_t_vec = (cos(gyr_norm * 0.5 * dt)*Eigen::Matrix4d::Identity() + 
              (1/gyr_norm)*sin(gyr_norm*0.5*dt)*omega)*q_vec;
       q2_t_vec = (cos(gyr_norm*0.25*dt)*Eigen::Matrix4d::Identity() + 
              (1/gyr_norm)*sin(gyr_norm*0.25*dt)*omega)*q_vec;
    }
    else{
       //Use limit version of equation
       q_t_vec = (Eigen::Matrix4d::Identity() + 0.5*dt*omega)*q_vec;
       q2_t_vec = (Eigen::Matrix4d::Identity() + 0.25*dt*omega)*q_vec;
    }

    q_t_dt = Eigen::Quaterniond(q_t_vec);
    q_t_dt2 = Eigen::Quaterniond(q2_t_vec);

    Eigen::Matrix3d C_dt_transpose = q_t_dt.normalized().toRotationMatrix().transpose();
    Eigen::Matrix3d C_dt2_transpose = q_t_dt2.normalized().toRotationMatrix().transpose();

    Eigen::Vector3d k_1_v = C_dt_transpose*acc + gravity;
    Eigen::Vector3d k_1_p = imu_vel;

    Eigen::Vector3d k_2_v = C_dt2_transpose*acc + gravity;
    Eigen::Vector3d k_2_p = imu_vel + k_1_v*(dt/2);

    Eigen::Vector3d k_3_v = C_dt2_transpose*acc + gravity;
    Eigen::Vector3d k_3_p = imu_vel + k_2_v*(dt/2);

    Eigen::Vector3d k_4_v = C_dt_transpose*acc + gravity;
    Eigen::Vector3d k_4_p = imu_vel + k_3_v*(dt/2);

    rotation_q = q_t_dt;
    rotation_q.normalize();
    imu_vel = imu_vel + (dt/6)*(k_1_v + 2*k_2_v + 2*k_3_v + k_4_v);
    imu_pos = imu_pos + (dt/6)*(k_1_p + 2*k_2_p + 2*k_3_p + k_4_p);

    return;
}

    //Maybe put in separate util file?
    Eigen::Matrix3d MSCK_Filter::skew_symmetric(const Eigen::Vector3d& vec){
        Eigen::Matrix3d vec_skew;
        vec_skew << 0, -vec(2), vec(1),
                    vec(2), 0, -vec(0),
                    -vec(1), vec(0), 0;
        return vec_skew;
    }

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

// void MSCK_Filter::Iterated_Extended_Kalman_Filter(){
//     //Jacobian Calculations
//     motion_model_Jacobian();
//     motion_noise_Jacobian();
//     observation_model_Jacobian();
//     observation_noise_Jacobian();
//     //Prediction Step
//     state_prediction();
//     covariance_prediction();
//     xo = xp;
//     int i = 0;
//     while (i<20){
//     xo = x;   
//     observation_model_Jacobian_O();
//     observation_noise_Jacobian_O();
//     //Kalman Gain
//     kalman_gain();
//     //Correction Step
//     covariance_correction_I();
//     state_correction_I();
//     i++;
//     }
//     }
// void MSCK_Filter::state_prediction(){
//        xp   << (x(0) + cos(x(2))*u(0)*dt),
//                (x(1) + sin(x(2))*u(0)*dt),
//                (x(2) + u(1)*dt);
//        xp(2) = std::fmod(xp(2) + M_PI, 2.0 * M_PI) - M_PI;
// }
// void MSCK_Filter::covariance_prediction(){
//        Pp = F * P * F.transpose() + Qp;
// }
// void MSCK_Filter::kalman_gain(){
//        K = Pp * G.transpose() * (G * Pp * G.transpose() + Rp).inverse();
// }
// void MSCK_Filter::state_correction(){
//        yp   <<(sqrt((xp(0)*xp(0))+(xp(1))*(xp(1)))), 
//               (std::atan2(-xp(1),-xp(0)) - xp(2)); 
//            // Ensure yp(1) is in the range [-pi, pi]
//        if (yp(1) < -M_PI) {
//            yp(1) += 2 * M_PI;
//        } else if (yp(1) > M_PI) {
//            yp(1) -= 2 * M_PI;
//        }

//        std::cout << "yp " << yp(0) <<" , " << yp(1)<< std::endl;
//        std::cout << "y " << y(0) <<" , " << y(1)<< std::endl;
//        x = xp + K * (y - yp);
// }
// void MSCK_Filter::state_correction_I(){
//        yo   <<(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), 
//               (std::atan2(-xo(1),-xo(0)) - xo(2)); 
//            // Ensure yp(1) is in the range [-pi, pi]
//        if (yo(1) < -M_PI) {
//            yo(1) += 2 * M_PI;
//        } else if (yo(1) > M_PI) {
//            yo(1) -= 2 * M_PI;
//        }

//        std::cout << "yp " << yo(0) <<" , " << yo(1)<< std::endl;
//        std::cout << "y " << y(0) <<" , " << y(1)<< std::endl;
//        x = xp + K * (y - yo - G*(xp - xo));
// }
// void MSCK_Filter::covariance_correction(){
//        P = (I - K * G) * Pp;
// }
// void MSCK_Filter::covariance_correction_I(){
//        P = (I - K * G) * Pp;
// }
// void MSCK_Filter::motion_model_Jacobian(){
//    F   << 1, 0, (-sin(x(2))*u(0)*dt),
//           0, 1, (cos(x(2))*u(0)*dt),
//           0, 0, 1;
// }
// void MSCK_Filter::motion_noise_Jacobian(){
//    wp   <<  cos(x(2)), 0,
//             sin(x(2)), 0,
//             0,         1;
//    Qp = dt*dt * wp * Q * wp.transpose();
// }
// void MSCK_Filter::observation_model_Jacobian(){
//    G   << (x(0))/(sqrt((x(0)*x(0))+(x(1))*(x(1)))), (x(1))/(sqrt((x(0)*x(0))+(x(1))*(x(1)))), 0,
//           (-x(1))/(sqrt((x(0)*x(0))+(x(1))*(x(1)))), (x(0))/(sqrt((x(0)*x(0))+(x(1))*(x(1)))), -1;
// }
// void MSCK_Filter::observation_model_Jacobian_O(){
//    G   << (xo(0))/(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), (xo(1))/(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), 0,
//           (-xo(1))/(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), (xo(0))/(sqrt((xo(0)*xo(0))+(xo(1))*(xo(1)))), -1;
// }
// void MSCK_Filter::observation_noise_Jacobian(){
//    Rp = R;
// }
// void MSCK_Filter::observation_noise_Jacobian_O(){
//    Rp = R;
// }

// namespace Kalman
