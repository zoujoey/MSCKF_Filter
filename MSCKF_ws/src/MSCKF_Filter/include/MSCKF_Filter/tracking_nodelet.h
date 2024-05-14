#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include "types.hpp"
#include "feature_tracker/ImageFeatures.h"

/*
General sensor definitions.
sensor_type: camera
comment: VI-Sensor cam0 (MT9M034)

Sensor extrinsics wrt. the body-frame.
T_BS:
  cols: 4
  rows: 4
  data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]

Camera specific definitions.
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
*/

namespace feature_tracker {

class FeatureTracker /*: public nodelet::Nodelet*/ {
public:
    FeatureTracker(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
    FeatureTracker()
      : FeatureTracker(ros::NodeHandle(), ros::NodeHandle("~")) {}
    ~FeatureTracker(){}
    //void onInit() override;

    int threshold = 10;

    static bool compare_response(cv::KeyPoint first, cv::KeyPoint second);

private:
    void image_callback(const sensor_msgs::ImageConstPtr &msg);        
    void publish_features(const cv_bridge::CvImagePtr &cv_image, const std::vector<cv::KeyPoint> &keypoints);
    void draw_features(cv_bridge::CvImagePtr imagePtr, const std::vector<cv::KeyPoint> &keypoints);
    
    //Feature detection and tracking
    void perform_FAST(const cv::Mat &img, std::vector<cv::KeyPoint> &pts, int num_features, int grid_x, int grid_y, int threshold);
    void perform_detection(const std::vector<cv::Mat> &img_pyr, const cv::Mat &mask, std::vector<cv::KeyPoint> &keypoints, int grid_x, int grid_y, int min_px_dist, int num_features, int threshold);
    void perform_matching(const std::vector<cv::Mat> &prev_pyr, const std::vector<cv::Mat> &curr_pyr, std::vector<cv::KeyPoint> &prev_keypoints, std::vector<cv::KeyPoint> &curr_keypoints, std::vector<unsigned char> &mask_out);

    //Feature detection helpers
    cv::Point2f undistort_cv(const cv::Point2f &uv_dist, std::string distortion_model);
    Eigen::Vector2f undistort_f_radtan(const Eigen::Vector2f &uv_dist);
    Eigen::Vector2f undistort_f_fisheye(const Eigen::Vector2f &uv_dist);

    int generate_feature_id();

    cv::Matx33d cam_matrix;
    cv::Vec4d distortion_coefficients;

    int num_features;
    int min_px_dist;
    int pyr_levels;
    cv::Size win_size;

    //Previous and Current Images
    cv_bridge::CvImagePtr cam0_img_curr;
    cv_bridge::CvImagePtr cam0_img_prev;

    //Previous and Current Pyramids
    std::vector<cv::Mat> cam0_pyr_curr;
    std::vector<cv::Mat> cam0_pyr_prev;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    image_transport::Subscriber image_sub;
    image_transport::Publisher image_pub;
    ros::Publisher imagePub_;
    ros::Subscriber camera_info_sub;
    ros::Publisher features_pub;

    cv::Ptr<cv::FastFeatureDetector> fast;
    cv::Ptr<cv::BFMatcher> matcher;
    std::map<int, FeatureTrack> feature_tracks;

    GridConfig grid_config;
    CameraParameters camera_parameters;
    bool initialized_camera = false;
    double MATCH_RATIO = 0.65;
};

} // namespace feature_tracker
