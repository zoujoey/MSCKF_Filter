cmake_minimum_required(VERSION 3.0.2)
project(MSCKF_Filter)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
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

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime 
  )

###########
## Build ##
###########

include_directories(
  include/MSCKF_Filter
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  )

add_executable(MSCK_Filter
  src/main_MSCKfilter.cpp
  src/MSCK_Filter.cpp
  src/tracking_nodelet.cpp
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
 
target_link_libraries(MSCK_Filter
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  )
