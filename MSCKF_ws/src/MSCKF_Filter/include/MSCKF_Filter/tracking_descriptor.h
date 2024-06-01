#ifndef TRACK_DESCRIPTOR_H
#define TRACK_DESCRIPTOR_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include "MSCKF_Filter/MSCK_Types.h"
#include "MSCKF_Filter/ImageFeatures.h"

namespace MSCKalman {

struct FeatureTrack {
    uint32_t latest_image_seq;
    std::vector<cv::KeyPoint> points;
};

class TrackDescriptor{
    public:
        TrackDescriptor(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
        TrackDescriptor() : TrackDescriptor(ros::NodeHandle(), ros::NodeHandle("~")) {}
        ~TrackDescriptor(){}

    private:
        //Callback, Publish, Drawing
        void image_callback(const sensor_msgs::ImageConstPtr &msg);   
        void publish_features(const cv_bridge::CvImagePtr &cv_image, const std::vector<cv::KeyPoint> &keypoints, const std::vector<size_t> ids);     
        void draw_features(cv_bridge::CvImagePtr imagePtr, const std::vector<cv::KeyPoint> &keypoints);
        void store_feature_tracks(const std::vector<cv::KeyPoint> &keypoints, const std::vector<size_t> ids, uint32_t image_seq);
        void draw_feature_tracks(cv::Mat &output_image);
        static bool compare_response(cv::KeyPoint first, cv::KeyPoint second);
        int generateFeatureID();

        //FAST feature detection
        void perform_FAST(const cv::Mat &img, cv::Mat mask, std::vector<cv::KeyPoint> &pts);
        void perform_detection(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::KeyPoint> &keypoints, cv::Mat &desc, std::vector<size_t> &ids);

        //Feature detection helpers
        cv::Point2f undistort_cv(const cv::Point2f &uv_dist, std::string distortion_model);
        Eigen::Vector2f undistort_f_radtan(const Eigen::Vector2f &uv_dist);
        Eigen::Vector2f undistort_f_fisheye(const Eigen::Vector2f &uv_dist);

        //Matching
        void robust_match(const std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, const cv::Mat &desc0,
                    const cv::Mat &desc1, std::vector<cv::DMatch> &matches);
        void robust_ratio_test(std::vector<std::vector<cv::DMatch>> &matches);
        void robust_symmetry_test(std::vector<std::vector<cv::DMatch>> &matches1, std::vector<std::vector<cv::DMatch>> &matches2,
                            std::vector<cv::DMatch> &good_matches);
        
        cv::Matx33d cam_matrix;
        cv::Vec4d distortion_coefficients;
        std::map<int, FeatureTrack> feature_tracks;

        //Orb Extractor
        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        // Our descriptor matcher
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

        // Parameters for our FAST grid detector
        int num_features;
        int threshold;
        int grid_x;
        int grid_y;
        int min_px_dist;
        double knn_ratio;
        size_t curr_id;

        //Variables
        std::vector<cv::KeyPoint> last_points;
        std::vector<size_t> last_ids;
        cv::Mat last_desc;
        cv::Mat last_img;
        cv::Mat last_mask;

        //descriptor matrix
        std::unordered_map<size_t, cv::Mat> desc_last;

        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;
        image_transport::Subscriber image_sub;
        image_transport::Publisher image_pub;
        ros::Publisher imagePub_;
        ros::Publisher features_pub;
};

}//namespace

#endif