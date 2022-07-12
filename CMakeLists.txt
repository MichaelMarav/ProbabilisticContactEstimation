cmake_minimum_required(VERSION 3.0.2)
project(friction_estimation)



find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_filters
)

find_package(Boost REQUIRED COMPONENTS
signals
)

catkin_package()
include_directories(
${catkin_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(friction_estimation src/friction.cpp)
add_dependencies(friction_estimation ${catkin_EXPORTED_TARGETS})
target_link_libraries(friction_estimation ${catkin_LIBRARIES} ${EIGEN3_LIBRARIES} ${Boost_LIBRARIES})



