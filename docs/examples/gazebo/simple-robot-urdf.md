---
title: "Simple Robot URDF Example"
sidebar_label: "Simple Robot URDF"
description: "Complete example of a simple differential drive robot URDF"
---

# Simple Robot URDF Example

## Overview

This example demonstrates how to create a simple differential drive robot using URDF (Unified Robot Description Format). The robot consists of a base body and two wheels, which is a common configuration for mobile robots.

## Complete URDF File

Create the file `simple_diff_drive.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_diff_drive" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.16"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.16"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0417" ixy="0.0" ixz="0.0" iyy="0.1042" iyz="0.0" izz="0.1458"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.04"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.000333" ixy="0.0" ixz="0.0" iyy="0.000333" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.04"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.000333" ixy="0.0" ixz="0.0" iyy="0.000333" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Castor Wheel (for stability) -->
  <link name="caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.18 -0.1" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.18 -0.1" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="0.2 0 -0.15" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo-specific elements -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.5</mu1>
    <mu2>0.5</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.5</mu1>
    <mu2>0.5</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="caster_wheel">
    <material>Gazebo/White</material>
    <mu1>0.1</mu1>
    <mu2>0.1</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <!-- Transmission for ROS Control -->
  <transmission name="left_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <update_rate>30</update_rate>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.36</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_wheel_tf>true</publish_wheel_tf>
      <publish_odom_tf>true</publish_odom_tf>
      <ros>
        <namespace>/</namespace>
      </ros>
    </plugin>
  </gazebo>
</robot>
```

## Creating a URDF Package

### 1. Create the Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake simple_robot_description
```

### 2. Create Directory Structure

```bash
cd ~/ros2_ws/src/simple_robot_description
mkdir -p urdf meshes config launch
```

### 3. Move URDF to Package

Create `~/ros2_ws/src/simple_robot_description/urdf/simple_diff_drive.urdf` with the content above.

### 4. Create Xacro Version (Optional)

Create `~/ros2_ws/src/simple_robot_description/urdf/simple_diff_drive.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot name="simple_diff_drive" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Include the main URDF -->
  <xacro:include filename="simple_diff_drive.urdf"/>
</robot>
```

### 5. Update package.xml

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>simple_robot_description</name>
  <version>0.0.0</version>
  <description>Simple differential drive robot description</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>robot_state_publisher</depend>
  <depend>joint_state_publisher</depend>
  <depend>xacro</depend>

  <exec_depend>urdf</exec_depend>
  <exec_depend>rviz2</exec_depend>

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
project(simple_robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install files
install(DIRECTORY
  urdf
  meshes
  config
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Testing the URDF

### 1. Build the Package

```bash
cd ~/ros2_ws
colcon build --packages-select simple_robot_description
source install/setup.bash
```

### 2. Check URDF Syntax

```bash
# Check the URDF file for syntax errors
check_urdf ~/ros2_ws/src/simple_robot_description/urdf/simple_diff_drive.urdf
```

### 3. Visualize in RViz

Create a launch file to visualize the robot:

```python
# ~/ros2_ws/src/simple_robot_description/launch/display.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_rviz = LaunchConfiguration('use_rviz')

    robot_description = Command([
        'xacro ',
        os.path.join(
            get_package_share_directory('simple_robot_description'),
            'urdf',
            'simple_diff_drive.urdf.xacro'
        )
    ])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_gui': True}]
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        condition=IfCondition(use_rviz),
        arguments=['-d', os.path.join(
            get_package_share_directory('simple_robot_description'),
            'config',
            'display.rviz'
        )]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        robot_state_publisher,
        joint_state_publisher,
        rviz
    ])
```

### 4. Launch Visualization

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch simple_robot_description display.launch.py
```

## Gazebo Integration

### 1. Create Gazebo Launch File

Create `~/ros2_ws/src/simple_robot_description/launch/gazebo.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command([
        'xacro ',
        PathJoinSubstitution([
            get_package_share_directory('simple_robot_description'),
            'urdf',
            'simple_diff_drive.urdf.xacro'
        ])
    ])
    robot_description = {'robot_description': robot_description_content}

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    # Robot state publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ])
    )

    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_entity
    ])
```

### 2. Launch in Gazebo

```bash
# Terminal 1: Start Gazebo
source ~/ros2_ws/install/setup.bash
ros2 launch simple_robot_description gazebo.launch.py

# Terminal 2: Send commands to the robot (after Gazebo loads)
source ~/ros2_ws/install/setup.bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.5}}'
```

## Common Issues and Solutions

### 1. Robot Falls Through Ground

This usually indicates problems with collision geometry or inertial properties:

- Ensure collision geometry matches visual geometry
- Check that inertial values are reasonable
- Verify that the robot starts above the ground

### 2. Joints Don't Move

- Check that joint names match between URDF and controller
- Verify transmission configuration
- Ensure proper controller configuration

### 3. Robot Tips Over

- Check center of mass and inertial properties
- Verify wheel contact with ground
- Adjust friction parameters in Gazebo

## Best Practices

1. **Start Simple**: Begin with basic shapes before adding complex geometry
2. **Validate Early**: Check URDF syntax before simulation
3. **Use Xacro**: Parameterize for reusability
4. **Proper Inertial Values**: Calculate realistic mass and inertia
5. **Test Incrementally**: Add one component at a time
6. **Document**: Comment your URDF for future reference

This simple robot example provides a foundation that you can extend with additional sensors, more complex geometry, or custom plugins for your specific application.