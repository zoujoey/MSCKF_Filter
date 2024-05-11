#ifndef MSCKF_FILTER_H
#define MSCKF_FILTER_H

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Image.h>
#include <eigen3/Eigen/Dense>
#include "MSCKF_Filter/MSCK_Types.h"
#include "MSCKF_Filter/math_utils.h"
namespace MSCKalman {

class MSCKF_Filter {
public:
    MSCKF_Filter();
    ~MSCKF_Filter() {}
//List of Functions
    //INTIALIZATION
    void init();
    void gravity_bias_initialization();
    
    //IMU PROPAGATION
    void propagate_imu(double dt, const Eigen::Vector3d& acc_m, const Eigen::Vector3d& gyr_m);
    void imu_state_estimate(const double& dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc);
    
    //ADDING A CAMERA FRAME
    void add_camera_frame(ImageSeq image_seq);
    void state_augmentation();
    void covariance_augmentation();
    int nFromInternalSeq(InternalSeq seq) const;
    int nFromImageSeq(InternalSeq seq) const;
    InternalSeq addImageSeq(ImageSeq image_seq);

    //ADDING FEATURES
    void addFeatures(ImageSeq image_seq, FeatureList features);
    void feature_position_estimate();
    //UPDATE STEP    
    void MSCKF_Update(const VectorXd &r, const MatrixXd &H, const MatrixXd &R);
    void qrDecomposition(VectorXd &r, MatrixXd &H, MatrixXd &R);
    void updateState(const VectorXd &state_error);


//List of Variables
    //STATE VECTOR
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
    Eigen::Quaterniond cam_imu_q {1,0,0,0}; //Rotation of IMU to Camera
    Eigen::VectorXd cam_imu_pose {0,0,0}; //Translation of IMU to Camera
    Eigen::VectorXd cam_state; //CAMERA STATE VECTOR (Stores all Camera Poses Needed (up to N-max))
    Eigen::Quaterniond cam_q; //Most Recent Quaternion of Camera wrt to Global Frame
    Eigen::VectorXd cam_pos; //Most Recent Position of Camera wrt to Global Frame

    //STATE COVARIANCE
    Eigen::MatrixXd F; //Error State Jacobian
    Eigen::MatrixXd G; //Error State Noise Jacobian 
    Eigen::MatrixXd P; //TOTAL COVARIANCE MATRIX
    Eigen::MatrixXd Pii; //Covariance Matrix of IMU
    Eigen::MatrixXd Pic; //Correlation Matrix between IMU and Camera
    Eigen::MatrixXd Pcc; //Covariance Matrix of Camera
    
    //MEASUREMENT MODEL
    Eigen::VectorXd acc_m; //accelerometer reading
    Eigen::VectorXd gyr_m; //gyroscope reading
    Eigen::MatrixXd acc_skew; //skew matrix acclerometer
    Eigen::MatrixXd gyr_skew; //skew matrix gyroscope
    Eigen::VectorXd imu_noise; //measurement noise of IMU
    Eigen::MatrixXd Q_imu; //Covariance Matrix of IMU Noise
    Eigen::MatrixXd phi; //State Transition Matrix

    //FEATURE LISTS
    // map of features for frames currently in the state
    std::map<FeatureId, FeatureInstanceList> features_;

    // features for frames for which are not part of the state yet
    std::map<ImageSeq, FeatureList> pending_features_;

    // map of given image sequence number to internal sequence number
    std::map<ImageSeq, InternalSeq> image_seqs;
    
    InternalSeq next_camera_seq{0};
    InternalSeq last_features_seq{0};
    int min_track_length{5};

    //useful constants
    Eigen::MatrixXd I; //Identity Matrix 
    double imu_dt; //IMU Time Step
    double cam_dt; //Camera Time Step
    double imu_current_time;
    double imu_last_time;
    bool imu_first_data;
    int N; //num camera poses
    int N_max{30};




    // Accessors for parts of state and covariance
    const VectorXd& state() const {
        return x;
    }
    const MatrixXd& covariance() const {
        return P;
    }
    Eigen::Ref<VectorXd> imuState() {
        return x.head<16>();
    }
    Eigen::Ref<VectorXd> cameraState() {
        return x.segment(16, 7 * nCameraPoses());
    }
    Eigen::QuaternionMapd cameraQuaternion(int n) {
        return Eigen::QuaternionMapd{x.segment<4>(16 + 7 * n).data()};
    }
    Matrix3d cameraRotation(int n) const {
        return Matrix3d{Quaterniond{x.segment<4>(16 + 7 * n).data()}};
    }
    Eigen::Ref<Vector3d> cameraPosition(int n) {
        return x.segment<3>(16 + 7 * n + 4);
    }
    Eigen::QuaternionMapd quaternion() {
        return Eigen::QuaternionMapd{quaternion_vector().data()};
    }
    Eigen::Map<const Quaterniond> quaternion() const {
        return Eigen::Map<const Quaterniond>{quaternion_vector().data()};
    }
    Eigen::Ref<Eigen::Vector4d> quaternion_vector() {
        return x.head<4>();
    }
    Eigen::Ref<const Eigen::Vector4d> quaternion_vector() const {
        return Eigen::Ref<const Eigen::Vector4d>{x.segment<4>(0)};
    }
    Eigen::Ref<Vector3d> biasGyro() {
        return x.segment<3>(4);
    }
    Eigen::Ref<Vector3d> velocity() {
        return x.segment<3>(7);
    }
    Eigen::Ref<Vector3d> biasVel() {
        return x.segment<3>(10);
    }
    Eigen::Ref<Vector3d> position() {
        return x.segment<3>(13);
    }
    Eigen::Ref<ImuCovMatrix> imuCovariance() {
        return P.topLeftCorner<15, 15>();
    }
    Eigen::Ref<MatrixXd> cameraCameraCovariance() {
        const auto N = nCameraPoses();
        return P.bottomRightCorner(6 * N, 6 * N);
    }
    Eigen::Ref<MatrixXd> imuCameraCovariance() {
        const auto N = nCameraPoses();
        return P.topRightCorner(15, 6 * N);
    }
    Eigen::Ref<MatrixXd> cameraImuCovariance() {
        const auto N = nCameraPoses();
        return P.bottomLeftCorner(6 * N, 15);
    }

    // The number of camera poses currently in the filter state (N)
    int nCameraPoses() const { return (x.size() - 16) / 7; }

    // The current position of the camera pose with given image sequence number
    // Throw std::range_error if not in state
    int nFromImageSeq(ImageSeq seq) const;

    InternalSeq next_camera_seq{0};
    InternalSeq last_features_seq{0};
    int min_track_length{5};
};

} // namespace MSCKalman

#endif // MSCKF_FILTER_H