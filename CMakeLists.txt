cmake_minimum_required(VERSION 3.0.2)
project(ros_ai_report)

find_package(
  catkin 
  REQUIRED COMPONENTS
  rospy 
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS 
  rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
