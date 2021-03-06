#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"

namespace myslam
{

class MapPoint;
class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long           id_;
    double                  time_stamp_;
    SE3                     T_c_w_;
    Camera::Ptr             camera_;
    Mat                     color_, depth_;
    bool                    is_key_frame_;  // whether a  key-frame

public:
    Frame();
    Frame(long id, double time_stamp=0, SE3 T_c_w=SE3(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat());
    ~Frame();

    // factory function
    static Frame::Ptr createFrame();

    // find the depth in depth map
    double findDepth(const cv::KeyPoint & kp);

    // Get camera center
    Vector3d getCamCenter() const;

    void setPose(const SE3& t_c_w);

    // check if a point is in this frame
    bool isInFrame(const Vector3d & pt_world);
};

}

#endif // FRAME_H

