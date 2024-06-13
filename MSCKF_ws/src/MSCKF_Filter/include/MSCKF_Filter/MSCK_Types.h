#ifndef MSCK_TYPES_H
#define MSCK_TYPES_H

#include <Eigen/Eigen>

namespace MSCKalman {

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::AngleAxisd;

using PoseVector = Eigen::Matrix<double, 7, 1>;  // Quaternion followed by position

using ImuStateVector = Eigen::Matrix<double, 16, 1>;
using ImuErrorVector = Eigen::Matrix<double, 12, 1>;
using ImuCovMatrix = Eigen::Matrix<double, 15, 15>;
using ImageSeq = uint32_t;
using InternalSeq = uint32_t;
using FeatureId = int32_t;

struct CameraParameters {
    double fx;
    double fy;
    double cx;
    double cy;
};

// Feature measurement tied to a specific feature track
struct ImageFeature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FeatureId id;
    Vector2d point;
    uint32_t lifetime;
};

// Feature measurement tied to a specific image frame
struct FeatureInstance {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    InternalSeq seq;
    Vector2d point;
    uint32_t lifetime;
};
//List of Features with Eigen 
using FeatureList = std::vector<ImageFeature>;
using FeatureInstanceList = std::vector<FeatureInstance>;

// Use Eigen's allocator for std::vector of fixed-size Eigen types
// See http://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
// Note we don't include Eigen/StdVector as it is not needed in C++11
using VectorOfVector2d = std::vector<Vector2d, Eigen::aligned_allocator<Vector2d>>;
using VectorOfVector3d = std::vector<Vector3d, Eigen::aligned_allocator<Vector3d>>;
using VectorOfMatrix3d = std::vector<Matrix3d, Eigen::aligned_allocator<Matrix3d>>;

} // namespace MSCKalman

#endif // MSCK_TYPES_H
