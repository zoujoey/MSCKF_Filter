#include "MSCKF_Filter/MSCK_Feature.h"

namespace MSCKalman {

MeasurementModelCostFunction::MeasurementModelCostFunction(
    const VectorOfVector2d& measurements,
    const VectorOfMatrix3d& camera_rotations,
    const VectorOfVector3d& camera_positions)
    : measurements_(measurements),
      camera_rotations_(camera_rotations),
      camera_positions_(camera_positions) {
    const int num_residuals = measurements.size() * 2; // 2D residuals for each measurement
    set_num_residuals(num_residuals);

    // Set parameter block size for alpha, beta, rho
    mutable_parameter_block_sizes()->push_back(3); // Alpha, beta, rho
}

bool MeasurementModelCostFunction::Evaluate(double const* const* parameters,
                                            double* residuals,
                                            double** jacobians) const {
    const int num_measurements = measurements_.size();

    for (int i = 0; i < num_measurements; ++i) {
        const Vector2d& z = measurements_[i];
        const Matrix3d& C = camera_rotations_[i];
        const Vector3d& p = camera_positions_[i];

        // Calculate h vector
        Vector3d h = C * Vector3d(parameters[0][0], parameters[0][1], 1) + parameters[0][2] * p;

        // Compute predicted measurement
        Vector2d predicted_meas = h.head<2>() / h(2);

        // Compute residuals
        residuals[2 * i] = predicted_meas(0) - z(0);
        residuals[2 * i + 1] = predicted_meas(1) - z(1);

        // Compute Jacobians if requested
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            // Compute Jacobian of residuals w.r.t parameters
            jacobians[0][2 * i * 3] = C(0, 0) / h(2);
            jacobians[0][2 * i * 3 + 1] = C(0, 1) / h(2);
            jacobians[0][2 * i * 3 + 2] = -h(0) * C(0, 0) / (h(2) * h(2)) + h(1) * C(0, 1) / (h(2) * h(2));
            
            jacobians[0][(2 * i + 1) * 3] = C(1, 0) / h(2);
            jacobians[0][(2 * i + 1) * 3 + 1] = C(1, 1) / h(2);
            jacobians[0][(2 * i + 1) * 3 + 2] = -h(0) * C(1, 0) / (h(2) * h(2)) + h(1) * C(1, 1) / (h(2) * h(2));
        }
    }

    return true;
}

Vector3d estimateFeaturePosition(const VectorOfVector2d& measurements,
                                 const VectorOfMatrix3d& camera_rotation_estimates,
                                 const VectorOfVector3d& camera_position_estimates,
                                 VectorXd& residuals) {
    const auto M = measurements.size();
    assert(M == camera_rotation_estimates.size());
    assert(M == camera_position_estimates.size());
    assert(M > 1);
    
    // Initial Guess 
    const auto& m1 = measurements.front();
    const auto& m2 = measurements.back();
    const auto& R1 = camera_rotation_estimates.front();
    const auto& R2 = camera_rotation_estimates.back();
    const auto& p1 = camera_position_estimates.front();
    const auto& p2 = camera_position_estimates.back();
    const auto pos_guess = triangulateFromTwoCameraPoses(m1, m2, R2 * R1.inverse(), R1 * (p2 - p1));
    auto params = VectorXd{inverseDepthParams(pos_guess)};
    
    // Create a Ceres problem
    ceres::Problem problem;
    
    for (int i = 0; i < M; ++i) {
        auto* cost_function = new MeasurementModelCostFunction(measurements, camera_rotation_estimates, camera_position_estimates);

        problem.AddResidualBlock(cost_function,
                                 nullptr /* loss function */,
                                 &params[0], &params[1], &params[2]);
    }

    // Configure solver options
    ceres::Solver::Options options;
    // Configure solver options...
    
    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // Output the residuals
    residuals.resize(2 * M);
    //model(params, residuals);
    
    // Compute global position
    auto pos = inverseDepthParams(params);
    auto global_pos = Vector3d{R1.transpose() * pos + p1};
    
    return global_pos;
}

Vector3d triangulateFromTwoCameraPoses(const Vector2d &measurement1,
                                       const Vector2d &measurement2,
                                       const Matrix3d &rotation1to2,
                                       const Vector3d &translation1to2) {
    // Use the method in Clement's paper
    // Unit direction vectors from each camera centre to the feature
    Vector3d dir1, dir2;
    dir1 << measurement1, 1;
    dir2 << measurement2, 1;
    dir1.normalize();
    dir2.normalize();

    // Put in the form Ax = b
    MatrixXd A{3, 2};
    A << dir1, -rotation1to2.transpose() * dir2;
    const auto &b = translation1to2;

    // Calculate A\b
    Vector2d x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    // Return position in first frame
    return x(0) * dir1;
}

} // namespace MSCKalman