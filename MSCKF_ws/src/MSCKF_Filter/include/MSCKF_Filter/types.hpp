#ifndef TYPES_HPP
#define TYPES_HPP

namespace feature_tracker {

using FeatureId = int32_t;

struct FeatureTrack {
    uint32_t latest_image_seq;
    std::vector<cv::KeyPoint> points;
};

struct GridConfig {
    int grid_row; // number of rows
    int grid_col; // number of columns
    int grid_min_features;
    int grid_max_features;
};

// Feature data
struct FeatureData {
    FeatureId id;
    float response;
    int lifetime;
    cv::Point2f cam_point;
};

// Camera parameters
struct CameraParameters {
    double fx;
    double fy;
    double cx;
    double cy;
};

} // namespace feature_tracker

#endif // TYPES_HPP
