
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

namespace myslam
{

VisualOdometry::VisualOdometry():
    state_(INITIALIZAING), ref_(nullptr), curr_(nullptr), map_(new Map), num_lost_(0), num_inliers_(0)
{
    num_of_features_        = Config::get<int>("number_of_features");
    scale_factor_           = Config::get<double>("scale_factor");
    level_pyramid_          = Config::get<int>("level_pyramid");
    match_ratio_            = Config::get<float>("match_ratio");
    max_num_lost_           = Config::get<float>("max_num_lost");
    min_inliers_            = Config::get<int>("min_inliers");
    key_frame_min_rot       = Config::get<double>("keyframe_rotation");
    key_frame_min_trans     = Config::get<double>("keyframe_translation");
    orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
}

}
