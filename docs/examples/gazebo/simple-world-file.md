---
title: "Simple Gazebo World Example"
sidebar_label: "Simple Gazebo World"
description: "Creating a simple world file for Gazebo simulation with obstacles"
---

# Simple Gazebo World Example

## Overview

This example demonstrates how to create a simple Gazebo world file with basic obstacles. A world file defines the environment in which your robot will operate, including terrain, lighting, physics properties, and static objects.

## Basic World Structure

### Minimal World File

Here's the most basic Gazebo world file:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="minimal_world">
    <!-- Include default models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine -->
    <physics name="default_physics" default="0" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

## Complete World Example with Obstacles

Here's a more complete world file with various obstacles:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_obstacles_world">
    <!-- Include default models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="default_physics" default="0" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Scene configuration -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- Simple Box Obstacle -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <pose>0 0 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
            <specular>1 0 0 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Cylindrical Obstacle -->
    <model name="cylinder_obstacle">
      <pose>-2 1 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
            <specular>0 1 0 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Wall Obstacle -->
    <model name="wall_obstacle">
      <pose>0 -3 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>6 0.2 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>6 0.2 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Ramp Obstacle -->
    <model name="ramp_obstacle">
      <pose>3 2 0.25 0 0 0.5</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>1 1 0 1</ambient>
            <diffuse>1 1 0 1</diffuse>
            <specular>1 1 0 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Custom Ground Plane with Texture -->
    <model name="custom_ground">
      <pose>0 0 0 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>10 10</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>10 10</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## World File Components Explained

### 1. World Element
```xml
<world name="world_name">
```
Defines the root element of the world file.

### 2. Include Elements
```xml
<include>
  <uri>model://ground_plane</uri>
</include>
```
Includes standard models like ground plane and sun.

### 3. Physics Configuration
```xml
<physics name="default_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
</physics>
```
Configures the physics engine properties:
- `max_step_size`: Time step for physics simulation
- `real_time_factor`: Target simulation speed relative to real time
- `real_time_update_rate`: Updates per second

### 4. Scene Configuration
```xml
<scene>
  <ambient>0.4 0.4 0.4 1</ambient>
  <background>0.7 0.7 0.7 1</background>
  <shadows>true</shadows>
</scene>
```
Configures visual appearance of the world.

### 5. Model Elements
```xml
<model name="model_name">
  <pose>x y z roll pitch yaw</pose>
  <static>true</static>
  ...
</model>
```
Defines objects in the world:
- `pose`: Position and orientation (x, y, z, roll, pitch, yaw)
- `static`: Whether the object is fixed in place

## Creating a World Package

### 1. Create the Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake simple_worlds
```

### 2. Create Directory Structure

```bash
cd ~/ros2_ws/src/simple_worlds
mkdir -p worlds models launch
```

### 3. Move World File

Create `~/ros2_ws/src/simple_worlds/worlds/simple_obstacles.sdf` with the content above.

### 4. Create Launch File

Create `~/ros2_ws/src/simple_worlds/launch/simple_world.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Path to world file
    world_file = PathJoinSubstitution([
        get_package_share_directory('simple_worlds'),
        'worlds',
        'simple_obstacles.sdf'
    ])

    # Launch Gazebo with the world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )

    return LaunchDescription([
        gazebo
    ])
```

### 5. Update package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>simple_worlds</name>
  <version>0.0.0</version>
  <description>Simple Gazebo worlds for testing</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>gazebo_ros_pkgs</exec_depend>
  <exec_depend>gazebo_plugins</exec_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### 6. Update CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(simple_worlds)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install files
install(DIRECTORY
  worlds
  models
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Testing the World

### 1. Build the Package

```bash
cd ~/ros2_ws
colcon build --packages-select simple_worlds
source install/setup.bash
```

### 2. Launch the World

```bash
# Launch the world directly with Gazebo
gazebo ~/ros2_ws/src/simple_worlds/worlds/simple_obstacles.sdf

# Or use the launch file
ros2 launch simple_worlds simple_world.launch.py
```

## Advanced World Features

### 1. Custom Models

You can include custom models in your world:

```xml
<include>
  <name>my_custom_model</name>
  <pose>1 2 0 0 0 0</pose>
  <uri>model://my_custom_model</uri>
</include>
```

### 2. Plugins in World Files

Add plugins directly to the world:

```xml
<plugin name="world_plugin" filename="libgazebo_ros_init.so">
  <ros>
    <namespace>/gazebo</namespace>
  </ros>
</plugin>
```

### 3. Wind Effects

Add environmental effects:

```xml
<world>
  <!-- ... other elements ... -->
  <wind>
    <linear_velocity>0.5 0 0</linear_velocity>
  </wind>
</world>
```

## Best Practices

1. **Start Simple**: Begin with basic shapes before complex environments
2. **Use Static for Obstacles**: Mark static objects as static for better performance
3. **Organize Files**: Keep world files in a dedicated package
4. **Document Poses**: Comment on object positions for clarity
5. **Test Performance**: Monitor simulation speed with complex worlds
6. **Validate Syntax**: Ensure SDF files are properly formatted

## Troubleshooting Common Issues

### 1. Objects Falling Through Ground

- Check collision geometry definitions
- Verify object poses are above ground level
- Ensure physics engine is properly configured

### 2. Poor Performance

- Reduce number of complex objects
- Simplify collision geometry
- Adjust physics parameters (larger step size)

### 3. Models Not Loading

- Verify model URIs are correct
- Check that models exist in Gazebo's model path
- Ensure proper file permissions

This world file example provides a foundation for creating more complex environments for robot testing and development.