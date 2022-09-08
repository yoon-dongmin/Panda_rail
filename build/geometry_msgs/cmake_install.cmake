# Install script for directory: /ros_ws/src/geometry_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/ros_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/geometry_msgs/msg" TYPE FILE FILES
    "/ros_ws/src/geometry_msgs/msg/Accel.msg"
    "/ros_ws/src/geometry_msgs/msg/AccelStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/AccelWithCovariance.msg"
    "/ros_ws/src/geometry_msgs/msg/AccelWithCovarianceStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Inertia.msg"
    "/ros_ws/src/geometry_msgs/msg/InertiaStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Point.msg"
    "/ros_ws/src/geometry_msgs/msg/Point32.msg"
    "/ros_ws/src/geometry_msgs/msg/PointStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Polygon.msg"
    "/ros_ws/src/geometry_msgs/msg/PolygonStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Pose2D.msg"
    "/ros_ws/src/geometry_msgs/msg/Pose.msg"
    "/ros_ws/src/geometry_msgs/msg/PoseArray.msg"
    "/ros_ws/src/geometry_msgs/msg/PoseStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/PoseWithCovariance.msg"
    "/ros_ws/src/geometry_msgs/msg/PoseWithCovarianceStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Quaternion.msg"
    "/ros_ws/src/geometry_msgs/msg/QuaternionStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Transform.msg"
    "/ros_ws/src/geometry_msgs/msg/TransformStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Twist.msg"
    "/ros_ws/src/geometry_msgs/msg/TwistStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/TwistWithCovariance.msg"
    "/ros_ws/src/geometry_msgs/msg/TwistWithCovarianceStamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Vector3.msg"
    "/ros_ws/src/geometry_msgs/msg/Vector3Stamped.msg"
    "/ros_ws/src/geometry_msgs/msg/Wrench.msg"
    "/ros_ws/src/geometry_msgs/msg/WrenchStamped.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/geometry_msgs/cmake" TYPE FILE FILES "/ros_ws/build/geometry_msgs/catkin_generated/installspace/geometry_msgs-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/ros_ws/devel/include/geometry_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/ros_ws/devel/share/roseus/ros/geometry_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/ros_ws/devel/share/common-lisp/ros/geometry_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/ros_ws/devel/share/gennodejs/ros/geometry_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python2" -m compileall "/ros_ws/devel/lib/python2.7/dist-packages/geometry_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/ros_ws/devel/lib/python2.7/dist-packages/geometry_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/ros_ws/build/geometry_msgs/catkin_generated/installspace/geometry_msgs.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/geometry_msgs/cmake" TYPE FILE FILES "/ros_ws/build/geometry_msgs/catkin_generated/installspace/geometry_msgs-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/geometry_msgs/cmake" TYPE FILE FILES
    "/ros_ws/build/geometry_msgs/catkin_generated/installspace/geometry_msgsConfig.cmake"
    "/ros_ws/build/geometry_msgs/catkin_generated/installspace/geometry_msgsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/geometry_msgs" TYPE FILE FILES "/ros_ws/src/geometry_msgs/package.xml")
endif()

