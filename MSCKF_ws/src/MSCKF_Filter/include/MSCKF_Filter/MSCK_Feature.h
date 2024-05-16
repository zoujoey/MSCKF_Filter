#ifndef MSCK_FEATURE_HPP
#define MSCK_FEATURE_HPP

#include "MSCK_Feature.h"
#include "math_utils.h"
#include <ceres/ceres.h>

namespace MSCKalman {

class MeasurementModelCostFunction : public ceres::CostFunction {
public:
    MeasurementModelCostFunction(const VectorOfVector2d& measurements,
                                 const VectorOfMatrix3d& camera_rotations,
                                 const VectorOfVector3d& camera_positions);

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override;

private:
    const VectorOfVector2d& measurements_;
    const VectorOfMatrix3d& camera_rotations_;
    const VectorOfVector3d& camera_positions_;
};

Vector3d estimateFeaturePosition(const VectorOfVector2d& measurements,
                                 const VectorOfMatrix3d& camera_rotation_estimates,
                                 const VectorOfVector3d& camera_position_estimates,
                                 VectorXd& residuals);

 // namespace MSCKalman
Vector3d triangulateFromTwoCameraPoses(const Vector2d &measurement1,
                                       const Vector2d &measurement2,
                                       const Matrix3d &rotation1to2,
                                       const Vector3d &translation1to2);
}

#endif // MSCK_FEATURE_HPP