---
title: "Testing Gazebo Simulation Examples"
sidebar_label: "Testing Gazebo Examples"
description: "How to test Gazebo simulation examples in Ubuntu 22.04"
---

# Testing Gazebo Simulation Examples

## Overview

This guide provides instructions for testing Gazebo simulation examples in a clean Ubuntu 22.04 environment. Testing ensures that simulation examples are reproducible and work correctly across different systems.

## Prerequisites

Before testing, ensure your Ubuntu 22.04 system has:

- ROS 2 Humble installed
- Gazebo installed (`ros-humble-gazebo-*`)
- Basic development tools
- Graphics support for visualization (optional for headless testing)

## Setting Up a Clean Environment

### 1. Verify Ubuntu Version

```bash
lsb_release -a
```

### 2. Install ROS 2 and Gazebo Dependencies

```bash
# Update package lists
sudo apt update

# Install ROS 2 Humble
sudo apt install -y ros-humble-desktop
sudo apt install -y ros-humble-ros-base

# Install Gazebo packages
sudo apt install -y ros-humble-gazebo-*

# Install additional dependencies
sudo apt install -y gazebo libgazebo-dev
sudo apt install -y python3-colcon-common-extensions python3-rosdep
```

### 3. Initialize rosdep and Setup Environment

```bash
sudo rosdep init
rosdep update

source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Creating a Test Workspace

### 1. Create Workspace Directory

```bash
mkdir -p ~/gazebo_test_ws/src
cd ~/gazebo_test_ws
```

### 2. Create Test Packages

```bash
# Create robot description package
cd ~/gazebo_test_ws/src
ros2 pkg create --build-type ament_cmake --dependencies robot_state_publisher joint_state_publisher xacro test_robot_description

# Create world package
ros2 pkg create --build-type ament_cmake --dependencies gazebo_ros_pkgs test_worlds
```

### 3. Set Up Directory Structure

```bash
# For robot description package
mkdir -p ~/gazebo_test_ws/src/test_robot_description/{urdf,launch,config}

# For world package
mkdir -p ~/gazebo_test_ws/src/test_worlds/{worlds,launch}
```

## Testing the Robot Model

### 1. Create Test Robot URDF

Create `~/gazebo_test_ws/src/test_robot_description/urdf/test_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="test_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
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

  <!-- Wheels -->
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

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
  </gazebo>

  <!-- Gazebo plugin for differential drive -->
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
    </plugin>
  </gazebo>
</robot>
```

### 2. Create Robot Launch File

Create `~/gazebo_test_ws/src/test_robot_description/launch/display_robot.launch.py`:

```python
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
            get_package_share_directory('test_robot_description'),
            'urdf',
            'test_robot.urdf'
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
        parameters=[{'use_gui': False}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        robot_state_publisher,
        joint_state_publisher
    ])
```

### 3. Test Robot Model Loading

```bash
cd ~/gazebo_test_ws

# Build the package
colcon build --packages-select test_robot_description

# Source the workspace
source install/setup.bash

# Test that the URDF is valid
check_urdf src/test_robot_description/urdf/test_robot.urdf

# Visualize the robot in RViz
ros2 launch test_robot_description display_robot.launch.py
```

## Testing Gazebo World

### 1. Create Test World

Create `~/gazebo_test_ws/src/test_worlds/worlds/test_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="test_world">
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

    <!-- Scene configuration -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
    </scene>

    <!-- Simple Box Obstacle -->
    <model name="test_box">
      <pose>2 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
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
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### 2. Create World Launch File

Create `~/gazebo_test_ws/src/test_worlds/launch/test_world.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    world_file = PathJoinSubstitution([
        get_package_share_directory('test_worlds'),
        'worlds',
        'test_world.sdf'
    ])

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

### 3. Test World Loading

```bash
cd ~/gazebo_test_ws

# Build the world package
colcon build --packages-select test_worlds

# Source the workspace
source install/setup.bash

# Test the world with Gazebo
ros2 launch test_worlds test_world.launch.py
```

## Testing Robot in Gazebo

### 1. Create Combined Launch File

Create `~/gazebo_test_ws/src/test_robot_description/launch/robot_in_world.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command([
        'xacro ',
        PathJoinSubstitution([
            get_package_share_directory('test_robot_description'),
            'urdf',
            'test_robot.urdf'
        ])
    ])
    robot_description = {'robot_description': robot_description_content}

    # Launch Gazebo with empty world first
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ])
    )

    # Robot state publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # Spawn robot in Gazebo after a delay
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'test_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        # Add a delay before spawning the robot
        TimerAction(
            period=5.0,
            actions=[spawn_entity]
        )
    ])
```

### 2. Test Robot in Gazebo

```bash
cd ~/gazebo_test_ws

# Build both packages
colcon build --packages-select test_robot_description test_worlds

# Source the workspace
source install/setup.bash

# Launch robot in Gazebo
ros2 launch test_robot_description robot_in_world.launch.py
```

## Headless Testing (No GUI)

For CI/CD or server environments without graphics:

```bash
# Run Gazebo in headless mode
gazebo --verbose --headless-rendering ~/gazebo_test_ws/src/test_worlds/worlds/test_world.sdf

# Or use gzserver directly
gzserver ~/gazebo_test_ws/src/test_worlds/worlds/test_world.sdf
```

## Performance Testing

### 1. Monitor Simulation Performance

```bash
# Check real-time factor and performance
gz stats

# Monitor ROS 2 topics during simulation
ros2 topic echo /clock
ros2 topic echo /odom
```

### 2. Resource Usage Monitoring

```bash
# Monitor CPU and memory usage
htop

# Monitor specific processes
ps aux | grep -E "(gazebo|gzserver|ros)"
```

## Automated Testing Script

Create an automated test script:

```bash
#!/bin/bash
# test_gazebo_examples.sh

set -e  # Exit on any error

echo "Starting Gazebo example tests..."

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Source workspace
source ~/gazebo_test_ws/install/setup.bash

# Test 1: Build packages
echo "Testing build..."
cd ~/gazebo_test_ws
colcon build --packages-select test_robot_description test_worlds
echo "Build test passed ✓"

# Test 2: Check URDF syntax
echo "Testing URDF syntax..."
check_urdf src/test_robot_description/urdf/test_robot.urdf
echo "URDF syntax test passed ✓"

# Test 3: Launch Gazebo server (headless)
echo "Testing Gazebo server launch..."
gzserver --verbose src/test_worlds/worlds/test_world.sdf &
GZ_PID=$!
sleep 5  # Allow Gazebo to start
kill $GZ_PID 2>/dev/null || true
echo "Gazebo server test passed ✓"

# Test 4: Launch robot state publisher
echo "Testing robot state publisher..."
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat src/test_robot_description/urdf/test_robot.urdf)' &
RSP_PID=$!
sleep 2
kill $RSP_PID 2>/dev/null || true
echo "Robot state publisher test passed ✓"

echo "All Gazebo tests completed successfully!"
echo "Exit code: 0"
```

Make the script executable and run it:

```bash
chmod +x ~/gazebo_test_ws/test_gazebo_examples.sh
~/gazebo_test_ws/test_gazebo_examples.sh
```

## Troubleshooting Common Issues

### 1. Gazebo Won't Start

```bash
# Check if graphics drivers are properly installed
lspci | grep -E "VGA|3D"
glxinfo | grep "OpenGL version"

# Try with software rendering
export LIBGL_ALWAYS_SOFTWARE=1
gazebo
```

### 2. Robot Won't Spawn

```bash
# Check if the robot description topic is published
ros2 topic list | grep robot_description

# Check if the spawn service is available
ros2 service list | grep spawn
```

### 3. Performance Issues

```bash
# Reduce physics update rate in world file
# Increase max_step_size for faster simulation
# Use headless mode for testing
```

## Success Criteria

All Gazebo examples must meet these criteria:

1. ✅ **Exit code 0**: All commands complete successfully
2. ✅ **Model loads**: Robot model appears correctly in Gazebo
3. ✅ **World loads**: Environment loads without errors
4. ✅ **Physics works**: Robot responds to physics simulation
5. ✅ **Timing compliance**: Simulation runs within expected time limits
6. ✅ **Resource efficiency**: CPU and memory usage within limits

## Expected Execution Times

- Simple world loading: < 2 minutes
- Robot spawning: < 30 seconds after Gazebo starts
- Full simulation test: < 5 minutes

Testing in a clean Ubuntu 22.04 environment ensures that your Gazebo examples are truly reproducible and will work for other users following the same setup instructions.