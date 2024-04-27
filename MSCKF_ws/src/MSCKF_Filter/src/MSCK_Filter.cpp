#include "MSCK_Filter.h"

namespace MSCKalman {

MSCK_Filter::MSCK_Filter(const ros::NodeHandle &nh,
                                 const ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh) {
  imu_sub = nh_.subscribe<sensor_msgs::Imu>("imu0", 1, &MSCK_Filter::IMU_Callback, this);
  ROS_INFO("Subscribe to IMU");
  //infoSubCAM_ = nh_.subscribe("noisy_pose_topic_CAM", 1, &MSCK_Filter::odom_Callback, this);
  odom_pub = nh_.advertise<nav_msgs::Odometry>("imu_odom_pub", 1000);
  ROS_INFO("Publish to imu_odom");
}

void MSCK_Filter::init(){
  // Initialize Prior Estimates
  N = 0; //Number of camera poses
  x = Eigen::VectorXd::Zero(16+7*N); // State Posterior Vector
  x_imu = Eigen::VectorXd::Zero(16); //Imu State Vector
  rotation_q = Eigen::Quaterniond::Identity();
  rotation_matrix.resize(3,3);//Rotational Matrix of Quaternion
  gyr_bias = Eigen::VectorXd::Zero(3);//Gyrscope Bias
  imu_vel = Eigen::VectorXd::Zero(3);//Velocity Vector
  acc_bias = Eigen::VectorXd::Zero(3);//Accelerometer Bias
  imu_pos = Eigen::VectorXd::Zero(3);//Position Vector
  cam_state = Eigen::VectorXd::Zero(7*N);//Camera State Vector
  cam_q = Eigen::VectorXd::Zero(7*N);//Camera State Rotation Recent
  cam_pos = Eigen::VectorXd::Zero(7*N);//Camera State Position Recent

  gravity = Eigen::Vector3d::Zero(); //Initialize gravity vector, set value later
  is_gravity_init = false;

  imu_dt = 0;
  imu_current_time = 0;
  imu_last_time = 0;
  imu_first_data = true;

  double gyr_noise = 0.001;
  double acc_noise = 0.01;
  double gyr_bias = 0.001;
  double acc_bias = 0.01;

  double velocity_cov = 0.25;
  double gyr_bias_cov = 1e-4;
  double acc_bias_cov = 1e-2;
  double extrinsic_rotation_cov = 3.0462e-4;
  double extrinsic_translation_cov = 1e-4;

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
  for(int i = 3; i < 6; i++){
    Pii(i, i) =  gyr_bias_cov;
  }
  for(int i = 6; i < 9; i++){
    Pii(i, i) = velocity_cov;
  }
  for(int i = 9; i < 12; i++){
    Pii(i,i) = acc_bias_cov;
  }

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
  Q_imu.block<3,3>(0,0) = Eigen::Matrix3d::Identity()*gyr_noise;
  Q_imu.block<3,3>(3,3) = Eigen::Matrix3d::Identity()*acc_noise;
  Q_imu.block<3,3>(6,6) = Eigen::Matrix3d::Identity()*gyr_bias;
  Q_imu.block<3,3>(9,9) = Eigen::Matrix3d::Identity()*acc_bias;
  Q_imu.resize(12,12);
  phi = Eigen::MatrixXd::Zero(15,15);
}

void MSCK_Filter::IMU_Callback(const sensor_msgs::ImuConstPtr& IMU_Msg){
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

    imu_dt = imu_time - imu_last_time;
    imu_last_time = imu_time;
}

void MSCK_Filter::propagate_imu(double dt, const Eigen::Vector3d& acc_m, const Eigen::Vector3d& gyr_m){
    ROS_INFO("Starting Propagate IMU");
    Eigen::Vector3d acc = acc_m - acc_bias;
    Eigen::Vector3d gyr = gyr_m - gyr_bias;

    Eigen::Matrix3d rotation_matrix = rotation_q.normalized().toRotationMatrix();

    //Transition (F) and Noise (G) Matrix
    //We assume planet rotation is negligible (w_g = 0)
    ROS_INFO("F and G Blocks");

    F.block<3,3>(0,0) = -skew_symmetric(gyr);
    F.block<3,3>(0,3) = -Eigen::Matrix3d::Identity();
    F.block<3,3>(6,0) = -rotation_matrix.transpose()*skew_symmetric(acc);
    F.block<3,3>(6,9) = -rotation_matrix.transpose();
    F.block<3,3>(12,6) = Eigen::Matrix3d::Identity();

    F.resize(15,15);

    G.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
    G.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
    G.block<3,3>(6,6) = -rotation_matrix.transpose();
    G.block<3,3>(9,9) = Eigen::Matrix3d::Identity();

    G.resize(15,12);

    //ROS_INFO("Done F and G Blocks");

    Eigen::MatrixXd I_15 = Eigen::MatrixXd::Identity(15,15);

    //Phi (State Transition Matrix)
    //Approximate Matrix Exponential Using Taylor Series
    //3rd Order For Now

    //ROS_INFO("Starting State Transition");
    Eigen::MatrixXd F_2 = F*F;
    Eigen::MatrixXd F_3 = F*F*F;
    phi = I_15 + F*dt + (1/2)*F_2*dt*dt + (1/6)*F_3*dt*dt*dt;
    //Discrete Time Noise Covariance Matrix Qk
    Eigen::MatrixXd Qk = phi*G*Q_imu*G.transpose()*phi.transpose()*dt;
    //Covariance Propagation for IMU
    Eigen::MatrixXd Pii_1 = phi*Pii*phi.transpose()+Qk;
    //Total Covariance Propagation
    Eigen::MatrixXd P_1 = Eigen::MatrixXd::Zero(15, 15);

    int temp = 6*N;

    P.block<15,15>(0,0) = Pii_1;
    P.block(15, 0, temp, 15) = (P.block(15,0, temp, 15))*phi.transpose();
    P.block(0, 15, 15, temp) = phi*(P.block(0,15,15,temp));
    P_1 = (P + P.transpose())*0.5;
    P = P_1;

    ROS_INFO("Starting IMU State Estimate");

    imu_state_estimate(dt, gyr, acc);   

    //std::cout << "Position: x = " << imu_pos(0) << ", y = " << imu_pos(1) << ", z = " << imu_pos(2) << std::endl;

    publishOdom(); 
}

void MSCK_Filter::gravity_bias_initialization(){
    ROS_INFO("Starting Initialization");
    Eigen::Vector3d sum_gyr = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d imu_gravity = Eigen::Vector3d::Zero();

    ROS_INFO("Entering Loop");
    for(const auto& IMU_Msg : imu_init_buffer){
       Eigen::Vector3d temp_acc_m = Eigen::Vector3d::Zero();
       Eigen::Vector3d temp_gyr_m = Eigen::Vector3d::Zero();

       tf::vectorMsgToEigen(IMU_Msg.linear_acceleration, temp_acc_m);
       tf::vectorMsgToEigen(IMU_Msg.angular_velocity, temp_gyr_m);

       sum_acc += temp_acc_m;
       sum_gyr += temp_gyr_m;
    }
    ROS_INFO("Gyr bias and IMU gravity");
    gyr_bias = sum_gyr / imu_init_buffer.size();
    imu_gravity = acc_m / imu_init_buffer.size();
    
    ROS_INFO("Set gravity");
    gravity(2) = -(imu_gravity.norm());

    std::cout << "Gyr Bias: " << gyr_bias << std::endl;
    std::cout << "Gravity: " << gravity << std::endl;

    //may have to adjust rotation initalization
    Eigen::Vector3d v = gravity.cross(-imu_gravity);
    double s = v.norm();
    double c = gravity.dot(-imu_gravity);
    Eigen::Matrix3d rotation_c = Eigen::Matrix3d::Identity() + skew_symmetric(v) + (skew_symmetric(v))*(skew_symmetric(v))*((1 - c) / s*s);
    ROS_INFO("Set rotation_q");
    rotation_q = Eigen::Quaterniond(rotation_c);
    ROS_INFO("Exit init");

}

void MSCK_Filter::imu_state_estimate(const double& dt, const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc){
    //IMU Quaternion Integration (0th Order)
    std::cout << "Linear Acc: " << acc << std::endl;
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

    //ROS_INFO("IMU Vel X: %d");
    std::cout << "dt: " << dt << std::endl;
    std::cout << "Position: x = " << imu_pos(0) << ", y = " << imu_pos(1) << ", z = " << imu_pos(2) << std::endl;
    std::cout << "Velocity: x = " << imu_vel(0) << ", y = " << imu_vel(1) << ", z = " << imu_vel(2) << std::endl;

    

    return;
}

void MSCK_Filter::publishOdom(){
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

//Maybe put in separate util file?
Eigen::Matrix3d MSCK_Filter::skew_symmetric(const Eigen::Vector3d& vec){
    Eigen::Matrix3d vec_skew;
    vec_skew << 0, -vec(2), vec(1),
                vec(2), 0, -vec(0),
                -vec(1), vec(0), 0;
    return vec_skew;
}

}