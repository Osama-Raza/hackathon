---
title: "Gazebo Simulation Setup"
sidebar_label: "Gazebo Simulation Setup"
description: "Setting up Gazebo simulation environment for robot development"
---

# Gazebo Simulation Setup

## Introduction to Gazebo

Gazebo is a powerful 3D simulation environment that enables accurate and efficient testing of robotics applications. It provides high-fidelity physics simulation, realistic rendering, and various sensors that allow you to test your robot code in a safe, virtual environment before deploying to real hardware.

## Installing Gazebo

### Prerequisites

Before installing Gazebo, ensure you have ROS 2 Humble installed on Ubuntu 22.04.

### Installation Steps

1. **Update package lists**:
```bash
sudo apt update
```

2. **Install Gazebo** (Gazebo Harmonic for ROS 2 Humble):
```bash
sudo apt install ros-humble-gazebo-*
```

This command installs all Gazebo-related ROS 2 packages, including:
- `ros-humble-gazebo-dev`: Development tools
- `ros-humble-gazebo-plugins`: Gazebo plugins for ROS 2
- `ros-humble-gazebo-ros`: ROS 2 interface for Gazebo
- `ros-humble-gazebo-ros-pkgs`: ROS 2 packages for Gazebo

3. **Install additional dependencies**:
```bash
sudo apt install gazebo libgazebo-dev
```

4. **Verify installation**:
```bash
gazebo --version
```

## Gazebo vs. Ignition Gazebo

ROS 2 Humble supports both traditional Gazebo (Gazebo Classic) and Ignition Gazebo (now called Ignition). For this curriculum, we'll focus on Gazebo Classic due to its wider adoption and documentation, though the concepts are transferable.

## Basic Gazebo Concepts

### Worlds

A world file defines the environment in which your robot operates. It includes:
- Terrain and static objects
- Lighting conditions
- Physics parameters
- Initial robot positions

### Models

Models represent objects in the simulation, including:
- Robots
- Obstacles
- Furniture
- Other dynamic objects

### Sensors

Gazebo provides various simulated sensors:
- Cameras
- LiDAR
- IMUs
- Force/torque sensors
- GPS
- Contact sensors

## Creating Your First Gazebo World

### Basic World Structure

Create a simple world file `my_empty_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Include the default sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Include the default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Scene configuration -->
    <scene>
      <ambient>0.8 0.8 0.8 1</ambient>
      <background>0.8 0.8 0.8 1</background>
    </scene>
  </world>
</sdf>
```

### Launching Gazebo with Your World

```bash
gazebo --verbose path/to/my_empty_world.sdf
```

## Integrating with ROS 2

### Launching Gazebo via ROS 2

You can launch Gazebo through ROS 2 launch files:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch Gazebo server
    gazebo_server = Node(
        package='gazebo_ros',
        executable='gzserver',
        arguments=[
            PathJoinSubstitution([
                get_package_share_directory('my_robot_gazebo'),
                'worlds',
                'my_empty_world.sdf'
            ])
        ],
        output='screen'
    )

    # Launch Gazebo client (GUI)
    gazebo_client = Node(
        package='gazebo_ros',
        executable='gzclient',
        output='screen'
    )

    return LaunchDescription([
        gazebo_server,
        gazebo_client
    ])
```

### Gazebo ROS 2 Control

Gazebo integrates with ROS 2 through plugins that enable:
- Joint state publishing
- Joint command execution
- Sensor data publishing
- Robot state management

## Common Gazebo Issues and Solutions

### 1. Performance Issues

If Gazebo runs slowly:
- Reduce physics update rate in world file
- Disable rendering (use `gzserver` instead of `gazebo`)
- Close other applications to free up resources

### 2. Model Loading Problems

If models don't load:
- Check that model files are in Gazebo's model path
- Verify model folder structure follows Gazebo conventions
- Ensure all required dependencies are installed

### 3. Plugin Issues

If plugins fail to load:
- Verify plugin libraries exist
- Check that plugin names match exactly
- Ensure proper permissions on plugin files

## Setting Up Gazebo Environment Variables

Add these to your `.bashrc` for convenience:

```bash
# Gazebo model path (add your custom models)
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/ros2_ws/src/my_robot_description/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/ros2_ws/src/my_robot_description/worlds

# Set Gazebo to use your local models first
export GAZEBO_MODEL_PATH=~/models:$GAZEBO_MODEL_PATH
```

## Testing Your Setup

### 1. Launch Gazebo with Default World

```bash
gazebo
```

You should see the Gazebo GUI with a default environment.

### 2. Spawn a Simple Model

```bash
# List available models
gz model --list

# If you have a model installed, try spawning it
gz model --spawn-file `gz model --info model://double_pendulum_with_base` --model-name double_pendulum
```

### 3. Check ROS 2 Integration

```bash
# List ROS 2 topics after starting Gazebo
ros2 topic list | grep gazebo

# Check if Gazebo services are available
ros2 service list | grep gazebo
```

## Troubleshooting Installation

### If Gazebo Fails to Launch

```bash
# Check for missing dependencies
sudo apt install --fix-missing

# Verify graphics drivers
lspci | grep -E "VGA|3D"

# Check OpenGL support
glxinfo | grep "OpenGL version"
```

### If Gazebo Runs but Has Rendering Issues

```bash
# Try software rendering
export LIBGL_ALWAYS_SOFTWARE=1
gazebo

# Or try with different graphics settings
gazebo --verbose --lockstep
```

## Performance Optimization

For better Gazebo performance:
- Use a dedicated GPU if available
- Increase system RAM if possible
- Use faster storage (SSD vs HDD)
- Close unnecessary applications
- Adjust physics parameters in world files

## Next Steps

With Gazebo properly installed and configured, you're ready to create robot models in URDF/SDF format and simulate their behavior. The next sections will cover robot description formats and how to create simulation-ready models.