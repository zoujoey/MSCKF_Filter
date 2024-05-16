#include "MSCK_Feature.h"  // Include the header file
#include <iostream>  // Include any necessary headers for testing
#include "MSCK_Types.h"
int main() {
    // Define test data
    // For simplicity, let's assume we have only two measurements,
    // two camera rotation estimates, and two camera position estimates
    MSCKalman::VectorOfVector2d measurements(2);
    measurements[0] << 10.0, 20.0;  // First measurement
    measurements[1] << 15.0, 25.0;  // Second measurement

    MSCKalman::VectorOfMatrix3d camera_rotations(2);
    // Fill camera_rotations with rotation matrices

    MSCKalman::VectorOfVector3d camera_positions(2);
    // Fill camera_positions with camera positions

    // Call the function under test
    Eigen::VectorXd residuals;
    Eigen::Vector3d estimated_position = MSCKalman::estimateFeaturePosition(
        measurements, camera_rotations, camera_positions, residuals);

    // Expected behavior:
    // Since this is just a simple test case, let's assume that the
    // estimateFeaturePosition function always returns a fixed value for simplicity.
    // You would typically compute the expected result based on the test data and the function implementation.
    Eigen::Vector3d expected_position(1.0, 2.0, 3.0);

    // Check if the output matches the expected result
    if (estimated_position.isApprox(expected_position)) {
        std::cout << "Test Passed: Estimated position matches expected result." << std::endl;
    } else {
        std::cerr << "Test Failed: Estimated position does not match expected result." << std::endl;
    }

    return 0;
}