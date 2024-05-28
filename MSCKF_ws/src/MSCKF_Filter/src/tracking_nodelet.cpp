#include "MSCKF_Filter/tracking_nodelet.h"

namespace MSCKalman {

    FeatureTracker::FeatureTracker(const ros::NodeHandle &nh, const ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh) {
        fast = cv::FastFeatureDetector::create();
        matcher = cv::BFMatcher::create();

        auto image_transport = image_transport::ImageTransport{nh_};
        features_pub = nh_.advertise<MSCKF_Filter::ImageFeatures>("features", 10);
        image_pub = image_transport.advertise("output_image", 1);
        image_sub = image_transport.subscribe("/cam0/image_raw", 3, &FeatureTracker::image_callback, this);
        imagePub_ = nh_.advertise<sensor_msgs::Image>("feature_image", 1000);

        cv::setNumThreads(4);

        num_features = 200;
        min_px_dist = 10;
        pyr_levels = 5;
        win_size = cv::Size(15,15);

        //Initialize camera matrix and distortion
        cam_matrix(0,0) = 458.654;
        cam_matrix(0,1) = 0;
        cam_matrix(0,2) = 367.215;
        cam_matrix(1,0) = 0;
        cam_matrix(1,1) = 457.296;
        cam_matrix(1,2) = 248.375;
        cam_matrix(2,0) = 0;
        cam_matrix(2,1) = 0;
        cam_matrix(2,2) = 1;

        distortion_coefficients(0) = -0.28340811;
        distortion_coefficients(1) = 0.07395907;
        distortion_coefficients(2) = 0.00019359;
        distortion_coefficients(3) = 1.76187114e-05;
    }

    void FeatureTracker::image_callback(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cam0_rgb;
        try{
            cam0_img_curr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
            cam0_rgb = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        }
        catch(cv_bridge::Exception& e){
            std::cout << "cv_bridge exception: " << e.what() << std::endl;
            return;
        }

        cv::Mat &image = cam0_img_curr->image;
        int grid_y = 5;
        int grid_x = 3;

        //Historgram equalize
        cv::Mat curr_img;
        cv::Mat prev_img;
        std::string histogram_method = "CLAHE";
    
        if(histogram_method == "HISTOGRAM"){
            cv::equalizeHist(image, curr_img);
        }
        else if(histogram_method == "CLAHE"){
            double eq_clip_limit = 10.0;
            cv::Size eq_win_size = cv::Size(8,8);
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
            clahe->apply(cam0_img_curr->image, curr_img);
        }
        else{
            curr_img = image;
        }

        std::vector<cv::KeyPoint> curr_keypoints;
        std::vector<cv::KeyPoint> prev_keypoints;
        cv::Mat descriptors;
        cv::Mat mask = cv::Mat::zeros(cv::Size(curr_img.cols, curr_img.rows), CV_8UC1);

        //extract image pyramid
        cv::buildOpticalFlowPyramid(curr_img, cam0_pyr_curr, win_size, pyr_levels); 
        //std::cout << "Pyr Size: " << cam0_pyr_curr.size() << std::endl;       

        std::vector<cv::KeyPoint> prev_good_points;
        if(!initialized_camera){
            std::vector<cv::KeyPoint> good_points;
            //std::cout << "Test Perf Detect" << std::endl;
            perform_detection(cam0_pyr_curr, mask, good_points, grid_x, grid_y, min_px_dist, num_features, threshold);
            //std::cout << "Initial Detect Done" << std::endl;
            prev_img = curr_img;
            cam0_pyr_prev = cam0_pyr_curr;
            prev_good_points = good_points;
            initialized_camera = true;
            return;
        }

        auto good_points_old = prev_good_points;
        //prev_good_points = good_points;   
        perform_detection(cam0_pyr_prev, mask, good_points_old, grid_x, grid_y, min_px_dist, num_features, threshold);

        for(auto &k : good_points_old){
            k.class_id = generate_feature_id();
        }

        //Display detected features on RVIZ
        //std::cout << "Num Features Detected: " << good_points_old.size() << std::endl;
        draw_features(cam0_rgb, good_points_old);
        imagePub_.publish(cam0_rgb->toImageMsg());

        //Track temporally
        std::vector<unsigned char> mask_out;
        std::vector<cv::KeyPoint> curr_good_points = good_points_old;
        perform_matching(cam0_pyr_prev, cam0_pyr_curr, good_points_old, curr_good_points, mask_out);
        //std::cout << "Num Matches: " << good_points_old.size() << std::endl;

        if(mask_out.empty()){
            prev_img = curr_img;
            cam0_pyr_prev = cam0_pyr_curr;
            initialized_camera = false;
            //std::cout << "Not enough points for RANSAC" << std::endl;
            return;
        }

        std::vector<cv::KeyPoint> good_points;

        //get good tracks, loop through all good points
        for(size_t i = 0; i < curr_good_points.size(); i++){
            //Guarantee points are non negative
            if(curr_good_points.at(i).pt.x < 0 || curr_good_points.at(i).pt.y < 0 || (int)curr_good_points.at(i).pt.x >= curr_img.cols ||
                (int)curr_good_points.at(i).pt.y >= curr_img.rows){
                continue;
            }
            //TODO: Check if in mask?
            if((int)mask.at<uint8_t>((int)curr_good_points.at(i).pt.y, (int)curr_good_points.at(i).pt.x) > 127){
                continue;
            }
            //Add to good points
            if(mask_out[i]){
                good_points.push_back(curr_good_points[i]);
            }
        }
        
        cv::Mat output = image;

        //Publish tracked points
        publish_features(cam0_img_curr, good_points);

        store_feature_tracks(good_points, cam0_img_curr->header.seq);

        cv::drawKeypoints(image, good_points, image, -1, cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

        draw_feature_tracks(output);

        image_pub.publish(cam0_img_curr->toImageMsg());

        prev_img = curr_img;
        cam0_pyr_prev = cam0_pyr_curr;
        prev_good_points = good_points;
    }

    void FeatureTracker::perform_FAST(const cv::Mat &img, std::vector<cv::KeyPoint> &pts, int num_features, int grid_x, int grid_y, int threshold){
        if(num_features < grid_x*grid_y){
            double ratio = (double)grid_x / (double)grid_y;
            grid_y = std::ceil(std::sqrt(num_features / ratio));
            grid_x = std::ceil(grid_y * ratio);
        }

        int num_features_grid = (int)((double)num_features / (double)(grid_x * grid_y)) + 1;

        int size_x = img.cols / grid_x;
        int size_y = img.rows / grid_y;

        //point extraction
        int ct_cols = std::floor(img.cols / size_x);
        int ct_rows = std::floor(img.rows / size_y);
        std::vector<std::vector<cv::KeyPoint>> collection(ct_cols * ct_rows);

        //parallel for loop 
        parallel_for_(cv::Range(0, ct_cols*ct_rows), [&](const cv::Range &range){
            for(int r = range.start; r < range.end; r++){
                //Calculate xy cell value
                int x = r % ct_cols * size_x;
                int y = r / ct_cols * size_y;

                //skip if out of bounds
                if(x + size_x > img.cols || y + size_y > img.rows){
                    continue;
                }

                //Calculate extraction region
                cv::Rect img_roi = cv::Rect(x, y, size_x, size_y);

                //Extract FAST features
                std::vector<cv::KeyPoint> pts_new;
                cv::FAST(img(img_roi), pts_new, threshold, true);

                //Sort pts_new by keypoint size
                std::sort(pts_new.begin(), pts_new.end(), FeatureTracker::compare_response);

                for(size_t i = 0; i < (size_t)num_features_grid && i < pts_new.size(); i++){
                    //create keypoint
                    cv::KeyPoint pt_cor = pts_new.at(i);
                    pt_cor.pt.x += (float)x;
                    pt_cor.pt.y += (float)y;

                    //reject if out of bounds
                    if((int)pt_cor.pt.x < 0  || (int)pt_cor.pt.x > img.cols || (int)pt_cor.pt.y < 0 || (int)pt_cor.pt.y > img.rows){
                        continue;
                    }

                    //check if in the mask region
                    // if(mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) > 127){
                    //     continue;
                    // }
                    collection.at(r).push_back(pt_cor);
                }
            }
        });

        //Combine all collections into single vector
        for(size_t r = 0; r < collection.size(); r++){
            pts.insert(pts.end(), collection.at(r).begin(), collection.at(r).end());
        }

        if(pts.empty()){
            return;
        }
        //Sub-pixel refine parameters
        cv::Size win_size = cv::Size(5, 5);
        cv::Size zero_zone = cv::Size(-1, -1);
        cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.001);

        //get vector of points
        std::vector<cv::Point2f> pts_refined;
        for(size_t i = 0; i < pts.size(); i++){
            pts_refined.push_back(pts.at(i).pt);
        }

        //get sub-pixel for all extracted features
        cv::cornerSubPix(img, pts_refined, win_size, zero_zone, term_crit);

        //save refined points
        for(size_t i = 0; i < pts.size(); i++){
            pts.at(i).pt = pts_refined.at(i);
        }
    }

    void FeatureTracker::perform_detection(const std::vector<cv::Mat> &img_pyr, const cv::Mat &mask, std::vector<cv::KeyPoint> &keypoints, int grid_x, int grid_y, int min_px_dist, int num_features, int threshold){
        cv::Size size_close((int)((float)img_pyr.at(0).cols / (float)min_px_dist),
                            (int)((float)img_pyr.at(0).rows / (float)min_px_dist));
        cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
        float size_x = (float)img_pyr.at(0).cols / (float)grid_x;
        float size_y = (float)img_pyr.at(0).rows / (float)grid_y;
        cv::Size size_grid(grid_x, grid_y);
        cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
        cv::Mat mask_updated = mask.clone();

        auto it = keypoints.begin();
        while(it != keypoints.end()){
            cv::KeyPoint kpt = *it;

            //Check if out of bounds
            int x = (int)kpt.pt.x;
            int y = (int)kpt.pt.y;
            int edge = 10;
            if(x < edge || x >= img_pyr.at(0).cols - edge || y < edge || y >= img_pyr.at(0).rows - edge){
                it = keypoints.erase(it);
                continue;
            }
            //calculate mask coordinates
            int x_close = (int)(kpt.pt.x / (float)min_px_dist);
            int y_close = (int)(kpt.pt.y / (float)min_px_dist);
            if(x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height){
                it = keypoints.erase(it);
                continue;
            }

            //Calculate grid cell
            int x_grid = std::floor(kpt.pt.x / size_x);
            int y_grid = std::floor(kpt.pt.y / size_y);
            if(x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height){
                it = keypoints.erase(it);
                continue;
            }

            //Check if keypoint is near another point
            if(grid_2d_close.at<uint8_t>(y_close, x_close) > 127){
                it = keypoints.erase(it);
                continue;
            }

            //check if in mask area
            if(mask.at<uint8_t>(y, x) > 127){
                it = keypoints.erase(it);
                continue;
            }

            //move forward to next point
            grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
            if(grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255){
                grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
            }

            //append to mask of image
            if(x - min_px_dist >= 0 && x + min_px_dist < img_pyr.at(0).cols && y - min_px_dist >= 0 && y + min_px_dist < img_pyr.at(0).rows){
                cv::Point pt1(x - min_px_dist, y - min_px_dist);
                cv::Point pt2(x + min_px_dist, y + min_px_dist);
                cv::rectangle(mask_updated, pt1, pt2, cv::Scalar(255), -1);
            }
            it++;
        }

        //compute how many more features needed to be extracted
        double min_feat_percent = 0.50;
        int num_feats_needed = num_features - (int)keypoints.size();
        if(num_feats_needed < std::min(20, (int)(min_feat_percent * num_features))){
            return;
        }

        std::vector<cv::KeyPoint> keypoints_ext;
        perform_FAST(img_pyr.at(0), keypoints_ext, num_features, grid_x, grid_y, threshold);

        //reject features that are close to a current feature
        std::vector<cv::KeyPoint> kpts_new;
        std::vector<cv::Point2f> pts_new;
        for(auto &kpt : keypoints_ext){
            int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
            int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
            if(x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height){
                continue;
            }
            //See if ythere is point at this location
            if(grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127){
                continue;
            }
            //add it otherwise
            kpts_new.push_back(kpt);
            pts_new.push_back(kpt.pt);
            grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
        }

        //Iterate through and only add valid points
        for(size_t i = 0; i < pts_new.size(); i++){
            kpts_new.at(i).pt = pts_new.at(i);
            keypoints.push_back(kpts_new.at(i));
        }
    }

    bool FeatureTracker::compare_response(cv::KeyPoint first, cv::KeyPoint second){
        return first.response > second.response;
    }

    void FeatureTracker::perform_matching(const std::vector<cv::Mat> &curr_pyr, const std::vector<cv::Mat> &prev_pyr, std::vector<cv::KeyPoint> &curr_keypoints, std::vector<cv::KeyPoint> &prev_keypoints, std::vector<unsigned char> &mask_out){
        assert(curr_keypoints.size() == prev_keypoints.size());

        if(curr_keypoints.empty() || prev_keypoints.empty()){
            return;
        }

        //Convert keypoints to points
        std::vector<cv::Point2f> curr_points, prev_points;
        for(size_t i = 0; i < curr_keypoints.size(); i++){
            curr_points.push_back(curr_keypoints.at(i).pt);
            prev_points.push_back(prev_keypoints.at(i).pt);
        }

        //Not enough points to use RANSAC, return empty
        if(curr_points.size() < 10){
            for(size_t i = 0; i < curr_points.size(); i++){
                mask_out.push_back((unsigned char)0);
            }
            return;
        }

        std::vector<unsigned char> mask_klt;
        std::vector<float> error;
        cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(curr_pyr, prev_pyr, curr_points, prev_points, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

        std::vector<cv::Point2f> curr_keypoints_undistorted, prev_keypoints_undistorted;

        std::string distortion_model = "radtan";
        for(size_t i = 0; i < curr_keypoints.size(); i++){
            curr_keypoints_undistorted.push_back(undistort_cv(curr_points.at(i), distortion_model));
            prev_keypoints_undistorted.push_back(undistort_cv(prev_points.at(i), distortion_model));
        }

        //RANSAC outlier rejection
        std::vector<unsigned char> mask_rsc;
        double max_focal_length_curr_img = std::max(cam_matrix(0,0), cam_matrix(1,1));
        double max_focal_length_prev_img = std::max(cam_matrix(0,0), cam_matrix(1,1));
        double max_focal_length = std::max(max_focal_length_curr_img, max_focal_length_prev_img);
        cv::findFundamentalMat(curr_keypoints_undistorted, prev_keypoints_undistorted, cv::FM_RANSAC, 2.0 / max_focal_length, 0.999, mask_rsc);

        for(size_t i = 0; i < mask_klt.size(); i++){
            auto mask = (unsigned char)((i < mask_klt.size() && mask_klt[i] && i < mask_rsc.size() && mask_rsc[i]) ? 1 : 0);
            mask_out.push_back(mask);
        }

        for(size_t i = 0; i < curr_points.size(); i++){
            curr_keypoints.at(i).pt = curr_points.at(i);
            prev_keypoints.at(i).pt = prev_points.at(i);
        }   
    }

    cv::Point2f FeatureTracker::undistort_cv(const cv::Point2f &uv_dist, std::string distortion_model){
        Eigen::Vector2f pt1, pt2;
        pt1 << uv_dist.x, uv_dist.y;

        if(distortion_model == "radtan"){
            pt2 = undistort_f_radtan(pt1);
        }
        else if(distortion_model == "fisheye"){
            pt2 = undistort_f_fisheye(pt1);
        }

        cv::Point2f pt_out;
        pt_out.x = pt2(0);
        pt_out.y = pt2(1);
        return pt_out;
    }

    Eigen::Vector2f FeatureTracker::undistort_f_radtan(const Eigen::Vector2f &uv_dist){
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0,0) = uv_dist(0);
        mat.at<float>(0,1) = uv_dist(1);
        mat = mat.reshape(2);
        
        cv::undistortPoints(mat, mat, cam_matrix, distortion_coefficients);

        Eigen::Vector2f pt_out;
        mat = mat.reshape(1);
        pt_out(0) = mat.at<float>(0,0);
        pt_out(1) = mat.at<float>(0,1);
        return pt_out;
    }

    Eigen::Vector2f FeatureTracker::undistort_f_fisheye(const Eigen::Vector2f &uv_dist){
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0,0) = uv_dist(0);
        mat.at<float>(0,1) = uv_dist(1);
        mat = mat.reshape(2);
        
        cv::fisheye::undistortPoints(mat, mat, cam_matrix, distortion_coefficients);

        Eigen::Vector2f pt_out;
        mat = mat.reshape(1);
        pt_out(0) = mat.at<float>(0,0);
        pt_out(1) = mat.at<float>(0,1);
        return pt_out;
    }

    int FeatureTracker::generate_feature_id(){
        static int id = 0;
        return id++;
    }

    
    void FeatureTracker::publish_features(const cv_bridge::CvImagePtr &cv_image, const std::vector<cv::KeyPoint> &keypoints){
        if(!initialized_camera){
            std::cout << "Have not recivied camera info" << std::endl;
        }
        MSCKF_Filter::ImageFeatures msg;
        msg.image_seq = cv_image->header.seq;

        //std::cout << "Published Feature Seq: " << cv_image->header.seq << std::endl;

        //std::cout << "Img Seq: " << cv_image->header.seq << std::endl;

        int i = 0; 
        for(const auto &k : keypoints){
            MSCKF_Filter::Feature f;
            f.position.x = k.pt.x;
            f.position.y = k.pt.y;
            f.id = k.class_id;
            f.response = k.response;
            //f.octave = k.octave;
            msg.features.push_back(f);
            ++i;
        }
        features_pub.publish(msg);
    }

    void FeatureTracker::draw_features(cv_bridge::CvImagePtr imagePtr, const std::vector<cv::KeyPoint> &keypoints){
        for(const auto &kpt : keypoints){
            cv::Point2f point = kpt.pt;
            cv::circle(imagePtr->image, point, 2.0, cv::Scalar(0, 255, 0), 2.0);
        }
    }

    void FeatureTracker::store_feature_tracks(const std::vector<cv::KeyPoint> &keypoints, uint32_t image_seq){
        for(const auto &k : keypoints){
            const auto &feature_id = k.class_id;

            feature_tracks[feature_id].latest_image_seq = image_seq;
            feature_tracks[feature_id].points.push_back(k);
        }
        for (auto it = feature_tracks.cbegin(); it != feature_tracks.cend();) {
            const auto &track = it->second;
            if (track.latest_image_seq != image_seq) {
                // This feature does not exist in the latest frame. Erase this track from the map
                it = feature_tracks.erase(it);
            } 
            else {
                ++it;
            }
        }
        //std::cout << "Num Feature Tracks: " << feature_tracks.size() << std::endl;
    }

    void FeatureTracker::draw_feature_tracks(cv::Mat &output_image){
        auto drawn = 0;
        auto longest_track = 0ul;

        std::cout << "Feature Tracks Size: " << feature_tracks.size() << std::endl;
        for (const auto &ft : feature_tracks) {
            const auto &points = ft.second.points;
            // change color based on track length
            std::cout << "Points Size: " << points.size()  << std::endl;

            auto color = cv::Scalar(255, 255, 0, 1);  // yellow
            auto i = points.size();
            if (i > 5) {
                color = cv::Scalar(50, 255, 50, 1);  // green
            } else {
                break;
            }

            for (i = 1; i < points.size(); ++i) {
                auto &curr = points[i];
                auto &prev = points[i - 1];
                arrowedLine(output_image, prev.pt, curr.pt, color);
            }
            if (points.size() > longest_track) {
                longest_track = points.size();
            }
            ++drawn;
        }
        std::cout << "Drew " << drawn << " tracks, the longest was: " << longest_track << std::endl;
    }

} //namespace