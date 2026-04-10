cmake_minimum_required(VERSION 3.8)
project(charuco_processor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)

# Executable

add_executable(charuco_processor src/charuco_processor.cpp)

target_include_directories(charuco_processor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(charuco_processor
  ${OpenCV_LIBRARIES})

ament_target_dependencies(charuco_processor
  rclcpp
  cv_bridge
  sensor_msgs)


# Install

install(TARGETS charuco_processor
  DESTINATION lib/${PROJECT_NAME})

ament_package()