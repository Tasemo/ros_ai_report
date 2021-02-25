cmake_minimum_required(VERSION 3.0.2)
project(ros_ai_report)

find_package(
  catkin 
  REQUIRED COMPONENTS
  message_generation
  rospy
  audio_capture
  std_msgs
  audio_common_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  CommandPrediction.srv
)

add_message_files(
  FILES
  AggregatedData.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  audio_common_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  rospy
  audio_capture
  std_msgs
  audio_common_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
  scripts/ai.py
  scripts/microphone.py
  scripts/processor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
