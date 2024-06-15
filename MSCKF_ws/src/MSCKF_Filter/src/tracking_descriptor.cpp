#include "MSCKF_Filter/tracking_descriptor.h"

namespace MSCKalman {
    TrackDescriptor::TrackDescriptor(const ros::NodeHandle &nh, const ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh) {
        auto image_transport = image_transport::ImageTransport{nh_};
        features_pub = nh_.advertise<MSCKF_Filter::ImageFeatures>("features", 3);
        image_pub = image_transport.advertise("output_image", 1);
        image_sub = image_transport.subscribe("/cam0/image_raw", 3, &TrackDescriptor::image_callback, this);
        imagePub_ = nh_.advertise<sensor_msgs::Image>("feature_image", 1000);

        cv::setNumThreads(4);

        num_features = 200;
        min_px_dist = 10;
        threshold = 10;
        grid_x = 3;
        grid_y = 5;
        knn_ratio = 0.70;
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

    void TrackDescriptor::draw_features(cv_bridge::CvImagePtr imagePtr, const std::vector<cv::KeyPoint> &keypoints){
        for(const auto &kpt : keypoints){
            cv::Point2f point = kpt.pt;
            cv::circle(imagePtr->image, point, 2.0, cv::Scalar(0, 255, 0), 2.0);
        }
    }

    void TrackDescriptor::publish_features(const cv_bridge::CvImagePtr &cv_image, const std::vector<cv::KeyPoint> &keypoints, const std::vector<size_t> ids){
        MSCKF_Filter::ImageFeatures msg;
        msg.image_seq = cv_image->header.seq;

        for(int i = 0; i < keypoints.size(); i++){
            const auto &k = keypoints[i];
            MSCKF_Filter::Feature f;
            f.position.x = k.pt.x;
            f.position.y = k.pt.y;
            f.id = ids[i];
            f.response = k.response;

            // Add the lifetime of the feature
            auto &track = feature_tracks[ids[i]];
            f.lifetime = track.frame_count;
            if (f.lifetime >= 4){
            // std::cout << "Feature ID: " << ids[i]+1 << " Lifetime: " << f.lifetime << " frames" << std::endl;
            }
            msg.features.push_back(f);
        }
        features_pub.publish(msg);
    }

    void TrackDescriptor::store_feature_tracks(const std::vector<cv::KeyPoint> &keypoints, const std::vector<size_t> ids, uint32_t image_seq){
        for(int i = 0; i < keypoints.size(); i++){
            const auto &k = keypoints[i];
            const auto &feature_id = ids[i];
            auto &track = feature_tracks[feature_id];
            if (track.points.empty()) {
                // This is a new track
                track.frame_count = 0;
            }
            track.latest_image_seq = image_seq;
            track.frame_count++; // Increment the frame count for this feature
            track.points.push_back(k);
        }

        // Print the lifetime of the features being removed
        for (auto it = feature_tracks.cbegin(); it != feature_tracks.cend();) {
            const auto &track = it->second;
            if (track.latest_image_seq != image_seq) {        
                it = feature_tracks.erase(it);
            } 
            else {
                ++it;
            }
        }
    }
    // void TrackDescriptor::store_feature_tracks(const std::vector<cv::KeyPoint> &keypoints, const std::vector<size_t> ids, uint32_t image_seq) {
    //     for (int i = 0; i < keypoints.size(); i++) {
    //         const auto &k = keypoints[i];
    //         const auto &feature_id = ids[i];

    //         auto &track = feature_tracks[feature_id];
    //         if (track.points.empty()) {
    //             // This is a new track
    //             track.frame_count = 0;
    //         }
    //         track.latest_image_seq = image_seq;
    //         track.frame_count++; // Increment the frame count for this feature
    //         track.points.push_back(k);
    //     }

    //     // Prepare to publish and delete old features
    //     std::vector<size_t> features_to_delete;
    //     for (auto it = feature_tracks.cbegin(); it != feature_tracks.cend();) {
    //         const auto &track = it->second;
    //         if (track.latest_image_seq != image_seq) {
    //             if (track.frame_count > 30) {
    //                 // Publish feature
    //                 MSCKF_Filter::ImageFeatures msg;
    //                 msg.image_seq = image_seq;

    //                 for (const auto &pt : track.points) {
    //                     MSCKF_Filter::Feature f;
    //                     f.position.x = pt.pt.x;
    //                     f.position.y = pt.pt.y;
    //                     f.id = it->first;
    //                     f.response = pt.response;
    //                     f.lifetime = track.frame_count;
    //                     msg.features.push_back(f);
    //                 }
    //                 features_pub.publish(msg);

    //                 // Mark for deletion
    //                 features_to_delete.push_back(it->first);
    //             }
    //             it = feature_tracks.erase(it);
    //         } else {
    //             ++it;
    //         }
    //     }

    //     // Delete the features marked for deletion
    //     for (const auto &id : features_to_delete) {
    //         feature_tracks.erase(id);
    //     }
    // }

    void TrackDescriptor::draw_feature_tracks(cv::Mat &output_image){
        auto drawn = 0;
        auto longest_track = 0ul;

        for (const auto &ft : feature_tracks) {
            const auto &points = ft.second.points;

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
    }

    void TrackDescriptor::image_callback(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cam0_rgb;
        cv_bridge::CvImagePtr cam0_img_curr;

        try{
            cam0_img_curr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
            cam0_rgb = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        }
        catch(cv_bridge::Exception& e){
            return;
        }

        cv::Mat &image = cam0_img_curr->image;

        cv::Mat img, mask;
        std::string histogram_method = "CLAHE";

        if(histogram_method == "HISTOGRAM"){
            cv::equalizeHist(image, img);
        }
        else if(histogram_method == "CLAHE"){
            double eq_clip_limit = 10.0;
            cv::Size eq_win_size = cv::Size(8,8);
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
            clahe->apply(cam0_img_curr->image, img);
        }
        else{
            img =  image;
        }
        mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1);

        //initialize tracking or have lost tracking
        if(last_points.empty()){
            std::vector<cv::KeyPoint> good_points;
            cv::Mat good_desc;
            std::vector<size_t> good_ids;
            perform_detection(img, mask, good_points, good_desc, good_ids);
            last_img = img;
            last_mask = mask;
            last_points = good_points;
            last_desc = good_desc;
            last_ids = good_ids;
            return;
        }

        std::vector<cv::KeyPoint> new_points;
        cv::Mat new_desc;
        std::vector<size_t> new_ids;

        perform_detection(img, mask, new_points, new_desc, new_ids);

        std::vector<cv::DMatch> matches_ll;
        robust_match(last_points, new_points, last_desc, new_desc, matches_ll);

        std::vector<cv::KeyPoint> good_points;
        cv::Mat good_desc;
        std::vector<size_t> good_ids;

        int num_track_last = 0;

        //adjust ids
        for(size_t i = 0; i < new_points.size(); i++){
            int idll = -1;
            for(size_t j = 0; j < matches_ll.size(); j++){
                if(matches_ll[j].trainIdx == (int)i){
                    idll = matches_ll[j].queryIdx;
                }
            }
            //replace current id with old id
            good_points.push_back(new_points[i]);
            good_desc.push_back(new_desc.row((int)i));
            if(idll != -1){
                good_ids.push_back(last_ids[idll]);
                num_track_last++;
            }
            else{
                good_ids.push_back(new_ids[i]);
            }
        }

        for(int i = 0; i < good_points.size(); i++){
            auto &k = good_points.at(i);
            k.class_id = good_ids[i];
        }

        publish_features(cam0_img_curr, good_points, good_ids);

        store_feature_tracks(good_points, good_ids, cam0_img_curr->header.seq);

        cv::Mat output = cam0_rgb->image;

        cv::drawKeypoints(output, good_points, output, -1, cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        
        draw_feature_tracks(output);

        image_pub.publish(cam0_rgb->toImageMsg());

        last_img = img;
        last_mask = mask;
        last_points = good_points;
        last_desc = good_desc;
        last_ids = good_ids;
    }


    void TrackDescriptor::perform_detection(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, std::vector<size_t> &ids){
        assert(keypoints.empty());

        //extract features with FAST
        std::vector<cv::KeyPoint> extracted_kpts;
        perform_FAST(img, mask, extracted_kpts);

        //Extract descriptors for each feature
        cv::Mat extracted_desc;
        this->orb->compute(img, extracted_kpts, extracted_desc);

        //2D Occupancy Grid for image
        cv::Size size((int)((float)img.cols / (float)min_px_dist), (int)((float)img.rows / (float)min_px_dist));
        cv::Mat grid_2d = cv::Mat::zeros(size, CV_8UC1);

        //For all good matches, append to vectors
        for(size_t i = 0; i < extracted_kpts.size(); i++){
            cv::KeyPoint kpt = extracted_kpts.at(i);
            int x = (int)kpt.pt.x;
            int y = (int)kpt.pt.y;
            int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
            int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
            if(x_grid < 0 || x_grid >= size.width || y_grid < 0 || y_grid >= size.height || x < 0 || x >= img.cols || y < 0 || y >= img.rows){
                continue;
            }
            //Check if keypoint is near another point
            if(grid_2d.at<uint8_t>(y_grid, x_grid) > 127){
                continue;
            }
            //Else, append keypoints and descriptors
            keypoints.push_back(extracted_kpts.at(i));
            descriptors.push_back(extracted_desc.row((int)i));
            //Handle IDs?
            size_t temp = ++curr_id;
            ids.push_back(temp);
            grid_2d.at<uint8_t>(y_grid, x_grid) = 255;
        }

        // for(auto &k : keypoints){
        //     k.class_id = generateFeatureID();
        // }
    }

    void TrackDescriptor::perform_FAST(const cv::Mat &img, cv::Mat mask, std::vector<cv::KeyPoint> &pts){
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
                std::sort(pts_new.begin(), pts_new.end(), TrackDescriptor::compare_response);

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
                    if(mask.at<uint8_t>((int)pt_cor.pt.y, (int)pt_cor.pt.x) > 127){
                        continue;
                    }
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

    bool TrackDescriptor::compare_response(cv::KeyPoint first, cv::KeyPoint second){
        return first.response > second.response;
    }

    void TrackDescriptor::robust_match(const std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, const cv::Mat &desc0,
                    const cv::Mat &desc1, std::vector<cv::DMatch> &matches){
        std::vector<std::vector<cv::DMatch>> matches0to1, matches1to0;

        //Match descriptors
        matcher->knnMatch(desc0, desc1, matches0to1, 2);
        matcher->knnMatch(desc1, desc0, matches1to0, 2);

        //Ratio test for both matches
        robust_ratio_test(matches0to1);
        robust_ratio_test(matches1to0);

        //Symmetry test
        std::vector<cv::DMatch> matches_good;
        robust_symmetry_test(matches0to1, matches1to0, matches_good);

        //Convert points for ransac
        std::vector<cv::Point2f> pts0_rsc, pts1_rsc;
        for(size_t i = 0; i < matches_good.size(); i++){
            int index_pt0 = matches_good.at(i).queryIdx;
            int index_pt1 = matches_good.at(i).trainIdx;
            //Push back the 2d point
            pts0_rsc.push_back(pts0[index_pt0].pt);
            pts1_rsc.push_back(pts1[index_pt1].pt);
        }
        
        //return if not enough points
        if(pts0_rsc.size() < 10){
            return;
        }

        //Normalize points
        std::vector<cv::Point2f> pts0_n, pts1_n;
        std::string distortion_model = "radtan";
        for(size_t i = 0; i < pts0_rsc.size(); i++){
            pts0_n.push_back(undistort_cv(pts0_rsc.at(i), distortion_model));
            pts1_n.push_back(undistort_cv(pts1_rsc.at(i), distortion_model));
        }

        //RANSAC outlier rejection
        std::vector<unsigned char> mask_rsc;
        double max_focal_length_img0 = std::max(cam_matrix(0,0), cam_matrix(1,1));
        double max_focal_length_img1 = std::max(cam_matrix(0,0), cam_matrix(1,1)); 
        double max_focal_length = std::max(max_focal_length_img0, max_focal_length_img1);
        cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, 1 / max_focal_length, 0.999, mask_rsc);

        //loop through good matches
        for(size_t i = 0; i < matches_good.size(); i++){
            if(mask_rsc[i] != 1){
                continue;
            }
            matches.push_back(matches_good.at(i));
        }

        //assign previous keypoint id for each match 
        // for(const auto &m : matches){
        //     pts1[m.queryIdx].class_id = pts0[m.trainIdx].class_id;
        // }
    }

    void TrackDescriptor::robust_ratio_test(std::vector<std::vector<cv::DMatch>> &matches){
        //loop through all matches
        for(auto &match : matches){
            if(match.size() > 1){
                if(match.size() > 1){
                    if(match[0].distance / match[1].distance > knn_ratio){
                        match.clear();
                    }
                }
                else{
                    match.clear();
                }
            }
        }
    }

    void TrackDescriptor::robust_symmetry_test(std::vector<std::vector<cv::DMatch>> &matches1, std::vector<std::vector<cv::DMatch>> &matches2,
                            std::vector<cv::DMatch> &good_matches){
        for(auto &match1 : matches1){
            if(match1.empty() || match1.size() < 2){
                continue;
            }
            for(auto &match2 : matches2){
                if(match2.empty() || match2.size() < 2){
                    continue;
                }
                if(match1[0].queryIdx == match2[0].trainIdx && match2[0].queryIdx == match1[0].trainIdx){
                    good_matches.emplace_back(cv::DMatch(match1[0].queryIdx, match1[0].trainIdx, match1[0].distance));
                    break;
                }
            }
        }
    }

    cv::Point2f TrackDescriptor::undistort_cv(const cv::Point2f &uv_dist, std::string distortion_model){
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

    Eigen::Vector2f TrackDescriptor::undistort_f_radtan(const Eigen::Vector2f &uv_dist){
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

    Eigen::Vector2f TrackDescriptor::undistort_f_fisheye(const Eigen::Vector2f &uv_dist){
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

    int TrackDescriptor::generateFeatureID(){
        static int id = 0;
        return id++;
    }

} //namespace