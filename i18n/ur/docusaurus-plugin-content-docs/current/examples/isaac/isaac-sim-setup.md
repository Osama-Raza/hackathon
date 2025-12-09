---
title: "Isaac Sim Setup Example"
sidebar_label: "Isaac Sim Setup"
description: "Example setup and configuration for NVIDIA Isaac Sim"
---

# Isaac Sim Setup Example

## Overview

This document provides a complete example of setting up NVIDIA Isaac Sim for robotics development. We'll cover installation, basic configuration, and a simple robot simulation example.

## System Requirements Check

Before installing Isaac Sim, verify your system meets the requirements:

```bash
# Check GPU capabilities
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check available memory
free -h

# Verify system information
cat /etc/os-release
lscpu | grep -E "(Thread|Core|Socket|Model name)"
```

## Installation via Docker (Recommended)

### 1. Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker
sudo apt update
sudo apt install -y docker.io

# Add current user to docker group
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker
```

### 2. Pull Isaac Sim Docker Image

```bash
# Pull the latest Isaac Sim image
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# Verify the image was pulled
docker images | grep isaac-sim
```

### 3. Run Isaac Sim Container

```bash
# Create directories for Isaac Sim data
mkdir -p ~/isaac-sim-cache ~/isaac-sim-logs ~/isaac-sim-config

# Run Isaac Sim with proper configuration
docker run --gpus all -it --rm \
  --network=host \
  --env "ACCEPT_EULA=Y" \
  --env "NVIDIA_VISIBLE_DEVICES=all" \
  --env "NVIDIA_DRIVER_CAPABILITIES=all" \
  --volume $HOME/isaac-sim-cache:/isaac-sim/cache/kit \
  --volume $HOME/isaac-sim-logs:/isaac-sim/logs \
  --volume $HOME/isaac-sim-config:/isaac-sim/config \
  --volume $HOME/isaac-sim-assets:/isaac-sim/assets \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --env "DISPLAY=$DISPLAY" \
  --env "QT_X11_NO_MITSHM=1" \
  --privileged \
  nvcr.io/nvidia/isaac-sim:4.2.0
```

## Alternative: Standalone Installation

### 1. Install Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install graphics drivers (if not already installed)
sudo apt install -y nvidia-driver-535

# Install CUDA (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-0

# Install other dependencies
sudo apt install -y python3-pip python3-venv build-essential
```

### 2. Download and Install Isaac Sim

```bash
# Create installation directory
mkdir -p ~/isaac-sim-install
cd ~/isaac-sim-install

# Download Isaac Sim (you'll need to register on NVIDIA Developer website first)
# Follow the download instructions from NVIDIA's website

# Extract and install
tar -xzf isaac-sim-*.tar.gz
cd isaac-sim-*

# Run the installation script
./isaac-sim.sh
```

## Basic Isaac Sim Configuration

### 1. Launch Isaac Sim

```bash
# For Docker installation, Isaac Sim should start automatically
# For standalone, launch from the installation directory:
./isaac-sim.sh
```

### 2. Basic Configuration

Once Isaac Sim is running, you can configure basic settings:

```python
# Example Python script to configure Isaac Sim settings
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view

# Set up the simulation world
my_world = World(stage_units_in_meters=1.0)

# Configure physics settings
my_world.scene.enable_gravity = True
my_world.set_physics_dt(1.0/60.0)  # 60 Hz physics update

# Set up a basic scene
assets_root_path = get_assets_root_path()
if assets_root_path is not None:
    # Add a ground plane
    my_world.scene.add_ground_plane("/World/defaultGroundPlane")

    # Add a simple cube
    cube_path = "/World/Cube"
    from omni.isaac.core.objects import DynamicCuboid
    my_world.scene.add(
        DynamicCuboid(
            prim_path=cube_path,
            name="my_cube",
            position=[0, 0, 1.0],
            size=0.5,
            mass=1.0
        )
    )

# Reset the world to apply changes
my_world.reset()

# Run simulation for a few steps
for i in range(100):
    my_world.step(render=True)
```

## Setting Up ROS 2 Bridge

### 1. Install ROS 2 Bridge

The ROS 2 bridge is typically included with Isaac Sim, but you may need to enable it:

```bash
# In Isaac Sim's Extension Manager (Window → Extensions):
# Enable Isaac ROS2 Bridge

# Or via command line in Isaac Sim's Python console:
import omni
omni.kit.extension.ExtensionManager.get_instance().set_extension_enabled("omni.isaac.ros2_bridge", True)
```

### 2. Test ROS 2 Connection

```bash
# In a separate terminal, check if ROS 2 topics are available
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash  # If using Isaac Sim Docker

# Check available topics
ros2 topic list

# Check available services
ros2 service list
```

## Simple Robot Example

### 1. Create a Robot Simulation Script

Create `robot_example.py`:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
import numpy as np
import carb

# Create world instance
my_world = World(stage_units_in_meters=1.0)

# Set up the scene
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not use Isaac Sim assets. Ensure Isaac Sim Nucleus server is running.")
else:
    # Add ground plane
    my_world.scene.add_ground_plane("/World/defaultGroundPlane")

    # Add a simple wheeled robot
    robot_path = "/World/Robot"
    add_reference_to_stage(
        usd_path=assets_root_path + "/Isaac/Robots/Turtlebot/turtlebot3_differential.usd",
        prim_path=robot_path
    )

    # Add the robot to the scene
    robot = Robot(prim_path=robot_path, name="turtlebot")
    my_world.scene.add(robot)

# Enable gravity
my_world.scene.enable_gravity = True

# Set physics timestep
my_world.set_physics_dt(1.0/60.0, substeps=4)

# Reset the world
my_world.reset()

# Main simulation loop
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_playing():
        # Example: Move robot forward
        if my_world.current_time_step_index % 100 == 0:
            # Get robot and apply some action
            robot_position, robot_orientation = robot.get_world_pose()
            print(f"Robot position: {robot_position}")

# Close the world
my_world.clear()
```

### 2. Run the Robot Example

```bash
# Save the script as robot_example.py
# Run it from Isaac Sim's Python console or as a standalone script

# In Isaac Sim's console:
exec(open('robot_example.py').read())
```

## Isaac Sim with PyTorch Integration

### 1. Install PyTorch in Isaac Sim Environment

```bash
# If using Docker, install in the container
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### 2. Deep Learning Example

Create `dl_robot_example.py`:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.types import ArticulationAction
import torch
import torch.nn as nn
import numpy as np

class SimplePolicy(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(SimplePolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return torch.tanh(self.network(x))

# Initialize world
my_world = World(stage_units_in_meters=1.0)

# Set up scene
assets_root_path = get_assets_root_path()
if assets_root_path:
    my_world.scene.add_ground_plane("/World/defaultGroundPlane")

    # Add wheeled robot
    robot = WheeledRobot(
        prim_path="/World/Robot",
        name="turtlebot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=assets_root_path + "/Isaac/Robots/Turtlebot/turtlebot3_differential.usd"
    )
    my_world.scene.add(robot)

# Initialize neural network policy
policy = SimplePolicy()
policy.eval()  # Set to evaluation mode

# Reset world
my_world.reset()

# Main loop with neural network control
for i in range(1000):
    my_world.step(render=True)

    if my_world.is_playing():
        # Get robot state (simplified example)
        position, orientation = robot.get_world_pose()
        velocity = robot.get_linear_velocity()

        # Create state vector [x_pos, y_pos, x_vel, y_vel]
        state = torch.tensor([position[0], position[1], velocity[0], velocity[1]],
                            dtype=torch.float32)

        # Get action from neural network
        with torch.no_grad():
            action = policy(state).numpy()

        # Apply action to robot (simplified - actual robot control may differ)
        robot.apply_wheel_actions(
            ArticulationAction(joint_indices=[0, 1],
                             positions=None,
                             velocities=action * 10.0)  # Scale for realistic movement
        )

# Clean up
my_world.clear()
```

## Isaac Sim Testing in Clean Environment

### 1. Create Test Script

Create `test_isaac_sim.sh`:

```bash
#!/bin/bash
# Test Isaac Sim installation and basic functionality

set -e  # Exit on any error

echo "Starting Isaac Sim tests..."

# Check if NVIDIA GPU is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU not detected"
    exit 1
fi
echo "✓ NVIDIA GPU detected"

# Check if Docker is available (for Docker installation)
if command -v docker &> /dev/null; then
    echo "✓ Docker is available"

    # Check if Isaac Sim image exists
    if docker images | grep -q "isaac-sim"; then
        echo "✓ Isaac Sim Docker image found"
    else
        echo "WARNING: Isaac Sim Docker image not found"
        echo "Run: docker pull nvcr.io/nvidia/isaac-sim:4.2.0"
    fi
else
    echo "INFO: Docker not found, assuming standalone installation"
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP "V\K\d+\.\d+")
    echo "✓ CUDA version: $CUDA_VERSION"
else
    echo "ERROR: CUDA not found"
    exit 1
fi

# Check available memory (recommended 32GB+ for Isaac Sim)
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$MEMORY_GB" -ge 32 ]; then
    echo "✓ Memory check passed: ${MEMORY_GB}GB available"
else
    echo "INFO: Memory check - ${MEMORY_GB}GB available (recommended: 32GB+)"
fi

# Check disk space
DISK_SPACE_GB=$(df -h $HOME | awk 'NR==2 {print $4}' | sed 's/[^0-9]*//g')
if [ "$DISK_SPACE_GB" -gt 50 ]; then
    echo "✓ Disk space sufficient: ${DISK_SPACE_GB}GB free"
else
    echo "WARNING: Limited disk space: ${DISK_SPACE_GB}GB free"
fi

echo "All basic checks passed!"
echo "Isaac Sim installation appears ready for use."
echo "Exit code: 0"
```

### 2. Make Test Script Executable and Run

```bash
chmod +x test_isaac_sim.sh
./test_isaac_sim.sh
```

## Troubleshooting Common Issues

### 1. GPU Not Detected

```bash
# Check GPU status
nvidia-smi

# Verify drivers
sudo apt install nvidia-driver-535
sudo reboot

# Check CUDA installation
ls /usr/local/cuda/
```

### 2. Isaac Sim Won't Start

```bash
# Check Docker permissions
sudo usermod -aG docker $USER
# Log out and log back in

# Check NVIDIA container runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Performance Issues

```bash
# Check if Isaac Sim is using GPU
nvidia-smi -l 1  # Monitor GPU usage

# Reduce graphics quality in Isaac Sim settings
# Go to Window → Stage Settings → Rendering Quality
```

## Isaac Sim Optimization

### 1. Performance Settings

```python
# Optimize Isaac Sim for better performance
import carb

# Set rendering quality (0=Low, 1=Medium, 2=High, 3=Production)
carb.settings.get_settings().set("/app/performace/level", 1)

# Reduce physics substeps for faster simulation
# In your simulation script:
my_world.set_physics_dt(1.0/30.0, substeps=2)  # Lower rate, fewer substeps
```

### 2. Memory Management

```bash
# Set Docker memory limits if needed
docker run --gpus all -it --rm --memory=32g \
  # ... other parameters
```

## Success Criteria

After completing the Isaac Sim setup:

1. ✅ **GPU Detected**: NVIDIA GPU with CUDA support is available
2. ✅ **Isaac Sim Launches**: The application starts without errors
3. ✅ **Basic Scene Loads**: Can create and view simple scenes
4. ✅ **Physics Works**: Objects respond to gravity and collisions
5. ✅ **ROS 2 Bridge**: Can communicate with ROS 2 if enabled
6. ✅ **Performance**: Runs at acceptable frame rates for your hardware

Isaac Sim provides a powerful platform for developing AI-powered robots with realistic physics simulation and advanced rendering capabilities.