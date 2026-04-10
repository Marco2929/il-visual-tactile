cmake_minimum_required(VERSION 3.8)
project(charuco_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)

# Add executable

add_executable(charuco_detector_node src/charuco_detector_node.cpp)

target_include_directories(charuco_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_compile_features(charuco_detector_node PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(
  charuco_detector_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
  tf2_ros
)

target_link_libraries(charuco_detector_node
  ${OpenCV_LIBRARIES}
)

# Install targets

install(TARGETS charuco_detector_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()