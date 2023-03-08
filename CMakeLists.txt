cmake_minimum_required(VERSION 3.0.2)
project(ProbabilisticContactEstimation)

add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  unitree_legged_msgs
  geometry_msgs
)

find_package(Boost REQUIRED COMPONENTS system)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ProbabilisticContactEstimaton
 CATKIN_DEPENDS roscpp rospy std_msgs unitree_legged_msgs
 DEPENDS system_lib
)


# message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
# if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
#   set(ARCH amd64)
# else()
#   set(ARCH arm64)
# endif()
# set(LEGGED_SDK_NAME -pthread libunitree_legged_sdk_${ARCH}.so lcm)

# set(EXTRA_LIBS ${LEGGED_SDK_NAME} lcm)


# set(CMAKE_CXX_FLAGS "-O3 -fPIC")

include_directories(
    include
    ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIR}
    ${unitree_legged_msgs_INCLUDE_DIRS}
    /home/despargy/go1_ws/src/unitree_legged_sdk/include
)
link_directories(/home/despargy/go1_ws/src/unitree_legged_sdk/lib)
link_directories($(catkin_LIB_DIRS) lib)

add_executable(slip_recovery src/slip_recovery.cpp)
target_link_libraries(slip_recovery ${EXTRA_LIBS} ${catkin_LIBRARIES})
add_dependencies(slip_recovery ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(init_imu_force src/init_imu_force.cpp)
target_link_libraries(init_imu_force ${EXTRA_LIBS} ${catkin_LIBRARIES})
add_dependencies(init_imu_force ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(check_forces src/check_forces.cpp)
target_link_libraries(check_forces ${EXTRA_LIBS} ${catkin_LIBRARIES})
add_dependencies(check_forces ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS src/go1_pce.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
