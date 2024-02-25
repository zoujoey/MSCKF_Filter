cmake_minimum_required(VERSION 3.0.2)
project(MSCK_Filter)

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
)

find_package(OpenCV REQUIRED)

catkin_package()

###########
## Build ##
###########

include_directories(
  include/MSCK_Filter
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(Noisy_Circular_Data_Generator
  src/ex1_MSCKFilter.cpp)

add_executable(MSCK_Filter
  src/main_MSCKfilter.cpp
  src/MSCK_Filter.cpp
)

target_link_libraries(Noisy_Circular_Data_Generator
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES})

target_link_libraries(MSCK_Filter
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES})