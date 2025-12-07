#!/bin/bash

# Script to provision Ubuntu 22.04 + ROS 2 Humble environment
# This script is designed to run on a clean Ubuntu 22.04 installation

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting ROS 2 Humble installation on Ubuntu 22.04..."

# Update package lists
sudo apt update

# Install locale settings
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 GPG key and repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package lists again
sudo apt update

# Install ROS 2 Humble packages
sudo apt install -y ros-humble-desktop
sudo apt install -y ros-humble-ros-base

# Install colcon build tool
sudo apt install -y python3-colcon-common-extensions

# Install ROS 2 dependencies
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init || echo "rosdep already initialized"
rosdep update

# Install additional useful packages for robotics development
sudo apt install -y python3-pip
sudo apt install -y python3-vcstool
sudo apt install -y gazebo libgazebo-dev
sudo apt install -y ros-humble-gazebo-*
sudo apt install -y ros-humble-navigation2
sudo apt install -y ros-humble-nav2-bringup
sudo apt install -y ros-humble-xacro
sudo apt install -y ros-humble-robot-state-publisher
sudo apt install -y ros-humble-joint-state-publisher
sudo apt install -y ros-humble-teleop-tools
sudo apt install -y ros-humble-joy

# Install Python packages commonly used in ROS 2 development
pip3 install -U argcomplete
pip3 install -U transforms3d

# Setup ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create a workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build --symlink-install

# Add workspace to bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

# Install VS Code extensions for ROS 2 development (optional)
echo "ROS 2 Humble installation completed successfully!"
echo "Please run 'source ~/.bashrc' or restart your terminal to use ROS 2."

# Verification commands
echo "To verify installation, run these commands after sourcing the environment:"
echo "  - ros2 --version"
echo "  - gazebo --version"
echo "  - colcon --version"