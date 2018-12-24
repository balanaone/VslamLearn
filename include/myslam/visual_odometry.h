#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam
{

class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING = -1,
        OK = 0,
        LOST
    };

    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and mop points

    Frame::Ptr  ref_;       // reference key-frame
    Frame::Ptr  curr_;      // current frame

    cv::Ptr<cv::ORB>        orb_;               // orb detector and computer
    vector<cv::Point3f>     pts_3d_ref_;        // 3d points in reference frame
    vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    Mat                     descriptors_curr_;  // descriptor in current frame
    Mat                     descriptors_ref_;   // descriptor in reference frame
    vector<cv::DMatch>      feature_matches_;   // feature matches
    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher

    SE3 T_c_r_estimated_;   // the estimated pose of current frame
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;          // number of lost times

    // parameters
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid;
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;     // ratio for selecting good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;        // minimum imliers

    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal transition for two key-frames
    double map_point_erase_ratio_;  // remove map point ratio

public: // functions
    VisualOdometry();
    ~VisualOdometry();

    bool addFrame(Frame::Ptr frame);        // add a new frame

protected:
    // inner operation
    void extractKeyPoints();
    void computeDescriptors();
    void featureMatching();
    void setRef3DPoints();
    void poseEstimationPnP();    

    void addKeyFrame();
    bool checkEstimatePose();
    bool checkKeyFrame();
};

}


#endif // VISUAL_ODOMETRY_H

