#include "MSCKF_Filter/MSCK_EKF.h"

namespace MSCKalman {

MSCKF_EKF::MSCKF_EKF(){
}

//INITIALIZE VARIABLES
void MSCKF_EKF::init() {
    // Initialize Prior Estimates
    //TODO Feature List of Stuff
    N = 0; //Number of camera poses
    x = Eigen::VectorXd::Zero(16+ (7*N)); // State Posterior Vector
    x_imu = Eigen::VectorXd::Zero(16); //Imu State Vector
    rotation_q = Eigen::Quaterniond::Identity();//Rotation Vector of IMU
    rotation_matrix.resize(3,3);//Rotational Matrix of Quaternion
    rotation_matrix = rotation_q.toRotationMatrix();
    gyr_bias = Eigen::VectorXd::Zero(3);//Gyrscope Bias
    imu_vel = Eigen::VectorXd::Zero(3);//Velocity Vector
    acc_bias = Eigen::VectorXd::Zero(3);//Accelerometer Bias
    imu_pos = Eigen::VectorXd::Zero(3);//Position Vector
    cam_imu_q = Eigen::Quaterniond::Identity();//FILL IN LATER - Camera to IMU Position
    cam_imu_pose = Eigen::VectorXd::Zero(3);//FILL IN LATER - Camera to IMU Translation
    cam_state = Eigen::VectorXd::Zero(7*N);//Camera State Vector
    cam_q = Eigen::Quaterniond::Identity();//Camera State Rotation Recent
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
    P = covariance();
    Pii.resize(15, 15);//Covariance Matrix of IMU
    Pii = Eigen::MatrixXd::Zero(15, 15);
    Pii = imuCovariance();
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
    //Adding Camera Frames and Feature Lists
}

//INTIALIZE GRAVITY BIAS 
void MSCKF_EKF::gravity_bias_initialization() {
    // Implement your gravity bias initialization function here
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

    //std::cout << "Gyr Bias: " << gyr_bias << std::endl;
    //std::cout << "Gravity: " << gravity << std::endl;

    //may have to adjust rotation initalization
    Eigen::Vector3d v = gravity.cross(-imu_gravity);
    double s = v.norm();
    double c = gravity.dot(-imu_gravity);
    Eigen::Matrix3d rotation_c = Eigen::Matrix3d::Identity() + skew_symmetric(v) + (skew_symmetric(v))*(skew_symmetric(v))*((1 - c) / s*s);
    ROS_INFO("Set rotation_q");
    rotation_q = Eigen::Quaterniond(rotation_c);
    ROS_INFO("Exit init");
}

//IMU PROPAGATION
void MSCKF_EKF::propagate_imu(double dt, const Eigen::Vector3d& acc_m, const Eigen::Vector3d& gyr_m) {
    // Implement your IMU propagation function here
    //ROS_INFO("Starting Propagate IMU");
    Eigen::Vector3d acc = acc_m - acc_bias;
    Eigen::Vector3d gyr = gyr_m - gyr_bias;

    Eigen::Matrix3d rotation_matrix = rotation_q.normalized().toRotationMatrix();

    //Transition (F) and Noise (G) Matrix
    //We assume planet rotation is negligible (w_g = 0)
    //ROS_INFO("F and G Blocks");

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

    P.block<15,15>(0,0) = Pii_1;
    P.block(15, 0, 6*N, 15) = (P.block(15,0, 6*N, 15))*phi.transpose();
    P.block(0, 15, 15, 6*N) = phi*(P.block(0,15,15,6*N));
    //Missing block?
    P_1 = (P + P.transpose())*0.5;
    P = P_1;

    //ROS_INFO("Starting IMU State Estimate");

    imu_state_estimate(dt, gyr, acc);   
    //std::cout << "Position: x = " << imu_pos(0) << ", y = " << imu_pos(1) << ", z = " << imu_pos(2) << std::endl;
}

void MSCKF_EKF::imu_state_estimate(const double& dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc) {
    // Implement your IMU state estimation function here
    //IMU Quaternion Integration (0th Order)
    //ROS_INFO("IMU Estimate");
    double gyr_norm = gyro.norm(); //get magnitude of gyro
    Eigen::Matrix4d omega = Eigen::Matrix4d::Zero();
    omega.block<3,3>(0,0) = -skew_symmetric(gyro);
    omega.block<3,1>(0,3) = gyro;
    omega.block<1,3>(3,0) = -gyro;

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
    /*
    std::cout << "dt: " << dt << std::endl;
    std::cout << "Position: x = " << imu_pos(0) << ", y = " << imu_pos(1) << ", z = " << imu_pos(2) << std::endl;
    std::cout << "Velocity: x = " << imu_vel(0) << ", y = " << imu_vel(1) << ", z = " << imu_vel(2) << std::endl;
    */
    return;
}

//ADD CAMERA FRAME
void MSCKF_EKF::add_camera_frame(ImageSeq image_seq){
    //ROS_INFO("Add Cam Frame");
    N = nCameraPoses();
    //std::cout << "N Size: " << N << std::endl;

    x.conservativeResize(x.size() + 7);
    state_augmentation();
    covariance_augmentation();

    //std::cout << "Seq2: " << image_seq << std::endl;

    // Add sequence number mapping
    const auto internal_seq = addImageSeq(image_seq);

    // Check if we already got the fAdeatures for this image
    if (pending_features_.count(image_seq)) {
        std::cout << "Remove: " << pending_features_.size() << std::endl;
        add_features(image_seq, pending_features_[image_seq]);
        pending_features_.erase(image_seq);
    }
}

int MSCKF_EKF::nFromInternalSeq(InternalSeq seq) const{
    const auto n = seq - next_camera_seq + nCameraPoses();
    if (n < 0 || seq >= next_camera_seq) {
        throw std::out_of_range("That camera pose is not in the filter state");
    }
    return n;
}

int MSCKF_EKF::nFromImageSeq(ImageSeq seq) const{
    return nFromInternalSeq(image_seqs.at(seq));
}

InternalSeq MSCKF_EKF::addImageSeq(ImageSeq image_seq){ 
    //std::cout << "Add Image Seq: " << image_seq << std::endl;
    image_seqs.insert({image_seq, next_camera_seq});
    return next_camera_seq++;
}

void MSCKF_EKF::state_augmentation() {
    // Implement your state augmentation function here
    //ROS_INFO("State Augmentation");
    cam_q = rotation_q*cam_imu_q;
    cam_pos = imu_pos + cam_q.toRotationMatrix().transpose()*cam_imu_pose;
    x.segment(16+7*N, 4)(0) = cam_q.x();
    x.segment(16+7*N, 4)(1) = cam_q.y();
    x.segment(16+7*N, 4)(2) = cam_q.z();
    x.segment(16+7*N, 4)(3) = cam_q.w();
    x.segment(16+7*N+4, 3)(0) = cam_pos.x();
    x.segment(16+7*N+4, 3)(1) = cam_pos.y();
    x.segment(16+7*N+4, 3)(2) = cam_pos.z();
    cam_state.resize(cam_state.size() + 7);
    cam_state = x.segment(16 + 7*N, 7);
}

void MSCKF_EKF::covariance_augmentation() {
    //ROS_INFO("Covariance Augment");
    // Implement your covariance augmentation function here
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6,6*N + 15);
    //top row
    J.block<3,3>(0,0) = cam_q.toRotationMatrix();
    J.block<3,9>(0,3) = Eigen::MatrixXd::Zero(3,9);
    J.block<3,3>(0,12) = Eigen::MatrixXd::Zero(3,3);
    J.block(0,15,3,6*N) = Eigen::MatrixXd::Zero(3,6*N);
    //bottom row
    J.block<3,3>(3,0) = skew_symmetric(cam_q.toRotationMatrix().transpose()*cam_imu_pose);
    J.block<3,9>(3,3) = Eigen::MatrixXd::Zero(3,9);
    J.block<3,3>(3,12) = Eigen::MatrixXd::Identity(3,3);
    J.block(3,15,3,6*N) = Eigen::MatrixXd::Zero(3,6*N);

    //Covariance Update
    Eigen::MatrixXd P_1 = P;
    Eigen::MatrixXd P_2 = Eigen::MatrixXd::Zero(21+6*N, 15+6*N);
    P_2.block(0,0,15+6*N,15+6*N) = Eigen::MatrixXd::Identity(15+6*N,15+6*N);
    P_2.block(15+6*N,0,J.rows(),J.cols()) = J;
    P.resize(21+6*N, 21+6*N);
    P = P_2*P_1*P_2.transpose();
}

//ADD FEATURES
void MSCKF_EKF::add_features(ImageSeq image_seq, FeatureList features/*const MSCKF_Filter::ImageFeaturesConstPtr &msg*/) {
    // Did we already have addCameraFrame() called with this image_seq?
    //std::cout << "Add Features Seq: " << image_seq << std::endl;
    //ROS_INFO("Add Features");
    if (image_seqs.count(image_seq)) {
        std::cout << "Check" << std::endl;
        for (auto &f : features) {
            last_features_seq = image_seqs[image_seq];
            features_[f.id].push_back(FeatureInstance{last_features_seq, f.point});
        }
        processFeatures();
    } else {
        // We got the features before the frame (it's possible). Keep for later.
        pending_features_[image_seq] = features;
    }
}

void MSCKF_EKF::processFeatures(){
    ROS_INFO("Process Features");
    const auto features_to_use = filterFeatures();
    if (features_to_use.empty()) {
        std::cout << "Empty" << std::endl;
        return;
    }
    std::cout << "Not Empty" << std::endl;
    // Get stacked, uncorrelated residuals and Jacobian
    auto r_o = VectorXd{};  // residuals
    auto H_o = MatrixXd{};  // measurement Jacobian
    auto R_o = MatrixXd{};  // measurement covariance
    estimate_feature_positions(features_to_use, r_o, H_o, R_o);

    if(use_qr_decomposition){
        qrDecomposition(r_o, H_o, R_o);
    }
    // Update the filter
    MSCKF_Update(r_o, H_o, R_o);
}

std::vector<FeatureInstanceList> MSCKF_EKF::filterFeatures() {
    ROS_INFO("Filter Features");
    auto features_to_use = std::vector<FeatureInstanceList>{};
    std::cout << "features_ Size: " << features_.size() << std::endl;
    for (auto it = features_.cbegin(); it != features_.cend();) {
        const auto &instances = it->second;
        if (isFeatureExpired(instances)) {
            //std::cout << "Expired" << std::endl;
            // This feature does not exist in the latest frame, therefore it has moved out of the frame.
            if (isFeatureUsable(instances) && features_to_use.size() < max_feature_tracks_per_update) {
                features_to_use.push_back(instances);
            }
            // Erase the feature from the list
            it = features_.erase(it);
        } else {
            ++it;
        }
    }
    return features_to_use;
}

bool MSCKF_EKF::isFeatureExpired(const FeatureInstanceList &instances) const {
    return (instances.back().seq < last_features_seq);
}

bool MSCKF_EKF::isFeatureUsable(const FeatureInstanceList &instances) const {
    return (instances.size() > min_track_length);
}

void MSCKF_EKF::estimate_feature_positions(const std::vector<FeatureInstanceList> &features,
                                   VectorXd &r_o,
                                   MatrixXd &H_o,
                                   MatrixXd &R_o) {
    // Implement your feature collapse function here
    ROS_INFO("Estimate Feature Positions");
    const auto total_rows = sizeOfNestedContainers(features);
    const auto N = nCameraPoses();
    const auto L = features.size();
    const auto n_residuals = 2 * total_rows - 3 * L;
    r_o.resize(n_residuals);  // residuals
    H_o.resize(n_residuals, 12 + 6 * N);  // measurement Jacobian

    auto total_i = 0;
    for (const auto &f : features) {
        // Collect the feature measurements and corresponding camera pose estimates
        // todo: seems inefficient
        const auto M = f.size();

        auto measurements = VectorOfVector2d{};
        auto camera_rotations = VectorOfMatrix3d{};
        auto camera_positions = VectorOfVector3d{};
        auto camera_indices = std::vector<int>{};

        for (const auto &instance : f) {
            measurements.push_back(instance.point);
            const auto n = nFromInternalSeq(instance.seq);
            camera_rotations.emplace_back(cameraQuaternion(n));
            camera_positions.emplace_back(cameraPosition(n));
            camera_indices.push_back(n);
        }

        auto residuals = VectorXd{2 * M};
        auto estimated_pos = estimateFeaturePosition(measurements, camera_rotations, camera_positions, residuals);
        auto estimated_local = Vector3d{camera_rotations.front() * (estimated_pos - camera_positions.front())};

        // Calculate Jacobians for this feature estimate
        auto H_X_j = MatrixXd{2 * M, 12 + 6 * N};
        auto H_f_j = MatrixXd{2 * M, 3};
        singleFeatureH(estimated_pos, camera_indices, H_X_j, H_f_j);

        // Get uncorrelated residuals
        projectLeftNullspace(H_f_j, residuals, H_X_j);

        // After that we expect the size to be smaller:
        const auto new_dim = 2 * M - 3;
        assert(new_dim == residuals.size());
        assert(new_dim == H_X_j.rows());

        // Push onto stacked matrices (todo: simplify)
        r_o.segment(total_i, new_dim) << residuals;
        H_o.block(total_i, 0, new_dim, 12 + 6 * N) << H_X_j;

        total_i += new_dim;
    }
}

//MSCKF_UPDATE
void MSCKF_EKF::MSCKF_Update(const VectorXd &r, const MatrixXd &H, const MatrixXd &R) {
    ROS_INFO("MSCKF Update");
    const auto l = r.size();
    const auto N = nCameraPoses();
    const auto d = 12 + 6 * N;
    assert(use_qr_decomposition || l == H.rows());
    assert(d == H.cols());
    assert(l == R.rows());
    assert(l == R.cols());

    // Calculate Kalman gain
    const auto K = MatrixXd{covariance_ * H.transpose() * (H * covariance_ * H.transpose() + R).inverse()};
    const auto state_change = VectorXd{K * r};

    // Note that state_change has rotations as 3 angles, while state has quaternions.
    // Thus we can't just add it to the state.
    assert(state_change.size() == state_.size() - 1 - N);
    assert(d == K.rows());
    assert(l == K.cols());

    // Update state estimate
    updateState(state_change);

    // Update covariance estimate
    covariance_ = (MatrixXd::Identity(d, d) - K * H) * covariance_ * (MatrixXd::Identity(d, d) - K * H).transpose()
            + K * R * K.transpose();
}

void MSCKF_EKF::projectLeftNullspace(const MatrixXd &H_f_j, VectorXd &r, MatrixXd &H_X_j) {
    ROS_INFO("Project Left Nullspace");
    // Implement your MSCKF update function here
    // As in Mourikis, A is matrix whose columns form basic of left nullspace of H_f.
    // Note left nullspace == kernel of transpose
    MatrixXd A = H_f_j.transpose().fullPivLu().kernel();
    H_X_j = (A.transpose() * H_X_j).eval();
    r = (A.transpose() * r).eval();
}


void MSCKF_EKF::singleFeatureH(const Vector3d &estimated_global_pos,
                         const std::vector<int> &cameraIndices,
                         MatrixXd &H_X_j,
                         MatrixXd &H_f_j) {
    ROS_INFO("Single Feature H");
    const auto M = cameraIndices.size();
    const auto N = nCameraPoses();

    H_X_j.setZero(2 * M, 12 + 6 * N);
    H_f_j.resize(2 * M, 3);

    for (auto i = 0 * M; i < M; ++i) {
        const auto &n = cameraIndices[i];
        const auto &C = cameraRotation(n);  // camera rotation estimate in global frame
        const auto &p = cameraPosition(n);  // camera position estimate in global frame
        const auto local_pos = Vector3d{C * (estimated_global_pos - p)};  // feature position estimate in camera frame

        // Fill in Jacobian block as given in Mourikis preprint (referenced in paper)
        auto J_i = Eigen::Matrix<double, 2, 3>{};
        J_i << 1, 0, -local_pos(0) / local_pos(2), 0, 1, -local_pos(1) / local_pos(2);
        J_i = J_i / local_pos(2);

        H_X_j.block<2, 6>(2 * i, 12 + 6 * n) << J_i * skew_symmetric(local_pos), -J_i * C;

        H_f_j.block<2, 3>(2 * i, 0) << J_i * C;
    }
}
void MSCKF_EKF::qrDecomposition(VectorXd &r, MatrixXd &H, MatrixXd &R) {
    ROS_INFO("QR Decomp");
    auto qr = H.householderQr();
    const auto m = H.rows();
    const auto n = H.cols();
    if (m == 0 || n == 0 || n > m) {
        return;
    }

    MatrixXd Q1 = qr.householderQ() * MatrixXd::Identity(m, n);
    H = qr.matrixQR().topLeftCorner(n, n).template triangularView<Eigen::Upper>();

    r = Q1.transpose() * r;
    R = Q1.transpose() * R * Q1;
}
void MSCKF_EKF::updateState(const VectorXd &state_error) {
    ROS_INFO("Update State");
    // The state holds quaternions but state_error should hold angle errors.
    const auto N = nCameraPoses();
    assert(state_error.size() == state_.size() - 1 - N);
    assert(state_error.size() == 12 + 6 * N);

    // Update quaternion multiplicatively
    auto dq = errorQuaternion(state_error.segment<3>(0));
    quaternion() = dq * quaternion();

    // Update rest of IMU state additively
    state_.segment<9>(4) += state_error.segment<9>(3);

    // Do the same for each camera state
    for (auto n = 0; n < N; ++n) {
        dq = errorQuaternion(state_error.segment<3>(12 + 6 * n));
        cameraQuaternion(n) = dq * cameraQuaternion(n);
        cameraPosition(n) += state_error.segment<3>(15 + 6 * n);
    }
}
} // namespace MSCKalman