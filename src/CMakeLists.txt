ADD_LIBRARY(myslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
    g2o_types.cpp
    visual_odometry.cpp
)

TARGET_LINK_LIBRARIES(myslam
    ${THIRD_PARTY_LIBS}
)
