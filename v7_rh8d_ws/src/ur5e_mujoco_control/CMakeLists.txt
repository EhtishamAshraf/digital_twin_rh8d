cmake_minimum_required(VERSION 3.8)
project(ur5e_mujoco_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(rh8d_mujoco_interfaces REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

set(COMMON_DEPS
  rclcpp
  moveit_ros_planning_interface
  trajectory_msgs
  control_msgs
  rclcpp_action
  moveit_msgs
  shape_msgs
  rh8d_mujoco_interfaces
)

# add executables
add_executable(gestures src/gestures.cpp)
add_executable(cartesian_pickNplace src/cartesian_pickNplace.cpp)
add_executable(object_pickNplace src/object_pickNplace.cpp)
add_executable(object_pick src/object_pick.cpp)

ament_target_dependencies(gestures
  ${COMMON_DEPS}
)
ament_target_dependencies(cartesian_pickNplace
  ${COMMON_DEPS}
)
ament_target_dependencies(object_pickNplace
  ${COMMON_DEPS}
)
ament_target_dependencies(object_pick
  ${COMMON_DEPS}
)

install(TARGETS
  gestures
  cartesian_pickNplace
  object_pickNplace
  object_pick
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()





