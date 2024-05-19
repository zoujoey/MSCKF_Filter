cmake_minimum_required(VERSION 3.0.2)
project(MSCKF_Filter)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(Eigen3)
find_package(OpenCV)
find_package(Ceres REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  nav_msgs
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  tf_conversions
  eigen_conversions
  message_generation
)

add_message_files(
  FILES
  Feature.msg
  ImageFeatures.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  #INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
  #LIBRARIES ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_library(msckf_filter
  src/MSCK_EKF.cpp 
  src/MSCK_Feature.cpp
  src/MSCK_Node.cpp
)
target_link_libraries(msckf_filter
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

add_library(feature_tracking
  src/tracking_nodelet.cpp
)
target_link_libraries(feature_tracking
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(feature_tracking_node
  src/tracking_node.cpp
)
target_link_libraries(feature_tracking_node
  feature_tracking
)

add_executable(msckf_main
  src/main_MSCKfilter.cpp
)
target_link_libraries(msckf_main
  msckf_filter
)