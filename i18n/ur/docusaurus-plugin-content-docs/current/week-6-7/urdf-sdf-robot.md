---
title: "URDF Robot Descriptions"
sidebar_label: "URDF Robot Descriptions"
description: "Understanding URDF and SDF for robot modeling in ROS 2 and Gazebo"
---

# URDF Robot Descriptions

## Introduction to Robot Description Formats

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) are XML-based formats used to describe robot models. URDF is primarily used in ROS for kinematic and dynamic properties, while SDF is used by Gazebo for simulation-specific properties.

## URDF (Unified Robot Description Format)

URDF is the standard format for describing robots in ROS. It defines the physical and visual properties of a robot, including links, joints, and their relationships.

### Basic URDF Structure

A minimal URDF file has this structure:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### Links

Links represent rigid bodies in the robot. Each link can have:

- **Visual**: How the link appears in visualization
- **Collision**: How the link interacts in collision detection
- **Inertial**: Physical properties for dynamics simulation

### Joints

Joints connect links and define their motion:

```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

Joint types include:
- `revolute`: Rotational joint with limits
- `continuous`: Rotational joint without limits
- `prismatic`: Linear sliding joint
- `fixed`: No movement (rigid connection)
- `floating`: 6DOF movement
- `planar`: Planar movement

### Complete URDF Example

Here's a more complete robot URDF example:

```xml
<?xml version="1.0"?>
<robot name="diff_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Material definitions -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.083" iyz="0.0" izz="0.133"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 -0.1" rpy="1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 -0.1" rpy="1.5707 0 0"/>
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
</robot>
```

## SDF (Simulation Description Format)

SDF is used by Gazebo for simulation-specific properties. While URDF is ROS-centric, SDF is Gazebo-centric.

### Basic SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.05</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.05</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>1.0 0.5 0.2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>1.0 0.5 0.2</size>
          </box>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
```

## Xacro: XML Macros for URDF

Xacro allows you to create parameterized URDF files using macros and variables:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_robot">
  <!-- Define properties -->
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="base_length" value="0.5"/>
  <xacro:property name="base_width" value="0.3"/>
  <xacro:property name="base_height" value="0.2"/>

  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix *joint_pose">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <xacro:insert_block name="joint_pose"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.083" iyz="0.0" izz="0.133"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="left">
    <origin xyz="0 ${base_width/2} 0" rpy="${pi/2} 0 0"/>
  </xacro:wheel>

  <xacro:wheel prefix="right">
    <origin xyz="0 -${base_width/2} 0" rpy="${pi/2} 0 0"/>
  </xacro:wheel>
</robot>
```

## Converting URDF to SDF

You can convert URDF to SDF using Gazebo tools:

```bash
# Convert URDF to SDF
gz sdf -p path/to/robot.urdf > robot.sdf

# Or using the older method
ros2 run xacro xacro path/to/robot.urdf.xacro > robot.urdf
gz sdf -p robot.urdf > robot.sdf
```

## Robot State Publisher

To publish robot joint states in ROS 2, use the robot state publisher:

```xml
<!-- In your launch file -->
<node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
  <param name="robot_description" value="$(find my_robot_description)/urdf/robot.urdf"/>
</node>
```

Or in Python launch file:

```python
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command([
        'xacro ',
        os.path.join(
            get_package_share_directory('my_robot_description'),
            'urdf',
            'robot.urdf.xacro'
        )
    ])

    robot_description = {'robot_description': robot_description_content}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    return LaunchDescription([
        node_robot_state_publisher
    ])
```

## Common URDF Issues and Solutions

### 1. Joint Limits

Always specify appropriate joint limits:

```xml
<limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
```

### 2. Inertial Properties

Calculate proper inertial properties for stable simulation:

```xml
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.083" iyz="0.0" izz="0.133"/>
</inertial>
```

### 3. Origin Definitions

Use proper origin definitions for joints:

```xml
<origin xyz="0.1 0 0" rpy="0 0 0"/>
```

## Validating URDF Files

### Check URDF Syntax

```bash
# Check URDF syntax
check_urdf path/to/robot.urdf

# If using xacro
ros2 run xacro xacro path/to/robot.urdf.xacro
```

### Visualize URDF

```bash
# Use rviz to visualize the robot
rviz2

# Or use the robot model display
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat path/to/robot.urdf)'
```

## Best Practices

1. **Use Xacro**: Parameterize your URDF files for reusability
2. **Proper Inertial Values**: Calculate realistic mass and inertia values
3. **Collision vs Visual**: Use simpler geometry for collision than visual
4. **Consistent Naming**: Use consistent naming conventions
5. **Documentation**: Comment your URDF files to explain the structure
6. **Validation**: Always validate your URDF before simulation

## Next Steps

With a proper robot description in URDF/SDF format, you're ready to integrate it with Gazebo simulation and control systems. The next sections will cover how to create simulation environments and integrate controllers with your robot models.