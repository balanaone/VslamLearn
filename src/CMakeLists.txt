ADD_LIBRARY(myslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
)

TARGET_LINK_LIBRARIES(myslam
    ${THIRD_PARTY_LIBS}
)
