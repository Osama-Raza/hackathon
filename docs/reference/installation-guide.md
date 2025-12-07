---
title: "Ubuntu 22.04 + ROS 2 Humble Installation Guide"
sidebar_label: "Ubuntu 22.04 + ROS 2 Installation"
description: "Complete guide for installing Ubuntu 22.04 LTS and ROS 2 Humble Hawksbill"
---

# Ubuntu 22.04 + ROS 2 Humble Installation Guide

## Overview

This guide provides step-by-step instructions for installing Ubuntu 22.04 LTS and ROS 2 Humble Hawksbill, the recommended environment for Physical AI and Robotics development. ROS 2 Humble is an LTS (Long Term Support) release with support until 2027, making it ideal for long-term projects.

## Prerequisites

Before beginning the installation, ensure you have:

- A computer with at least 8GB RAM (16GB+ recommended)
- 50GB+ of free disk space
- Stable internet connection
- Administrative (sudo) access to the system
- Basic familiarity with Linux command line

## Installing Ubuntu 22.04 LTS

### Option 1: Fresh Installation

1. **Download Ubuntu 22.04 LTS**
   - Visit https://releases.ubuntu.com/22.04/
   - Download the appropriate ISO file (Desktop or Server)

2. **Create Bootable USB Drive**
   ```bash
   # Using Rufus (Windows) or Etcher (Cross-platform)
   # Or from command line on Linux:
   sudo dd if=ubuntu-22.04.3-desktop-amd64.iso of=/dev/sdX bs=4M status=progress && sync
   ```

3. **Boot from USB and Install**
   - Restart computer and boot from USB drive
   - Follow the Ubuntu installation wizard
   - Choose "Normal installation" with updates and third-party software
   - Select timezone and keyboard layout
   - Create user account

### Option 2: Dual Boot with Existing OS

1. **Shrink Existing Partition**
   - Use GParted or Windows Disk Management to create free space
   - Recommend at least 50GB for Ubuntu partition

2. **Boot from Ubuntu USB**
   - Follow installation steps as above
   - Choose "Install Ubuntu alongside Windows" option

### Option 3: Virtual Machine

1. **Install Virtualization Software**
   - VirtualBox: `sudo apt install virtualbox virtualbox-ext-pack`
   - VMware Workstation (proprietary)
   - QEMU/KVM for advanced users

2. **Create Virtual Machine**
   - Allocate minimum 4GB RAM, 50GB disk space
   - Enable hardware virtualization in BIOS/UEFI
   - Install Ubuntu in VM following standard process

## System Preparation

After installing Ubuntu, update the system and install essential packages:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    wget \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    net-tools \
    htop \
    tmux \
    vim \
    terminator
```

## Installing ROS 2 Humble Hawksbill

### Step 1: Set up the ROS 2 apt Repository

```bash
# Add the ROS 2 GPG key
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Step 2: Install ROS 2 Packages

```bash
# Update package lists
sudo apt update

# Install ROS 2 Desktop (includes Gazebo, RViz, etc.)
sudo apt install -y ros-humble-desktop

# Install additional packages for development
sudo apt install -y \
    ros-humble-ros-base \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers
```

### Step 3: Install Python Packages

```bash
# Install Python packages for ROS 2 development
pip3 install -U \
    argcomplete \
    rosdep \
    vcstool

# Initialize rosdep
sudo rosdep init
rosdep update
```

### Step 4: Setup Environment

Add ROS 2 environment setup to your shell profile:

```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

For zsh users:
```bash
echo "source /opt/ros/humble/setup.zsh" >> ~/.zshrc
source ~/.zshrc
```

## Installing Development Tools

### Colcon (Build Tool)

```bash
# Install colcon and extensions
sudo apt install python3-colcon-common-extensions
```

### Development Environment Setup

```bash
# Install VS Code
wget -qO - https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install -y code

# Install useful VS Code extensions for ROS 2
code --install-extension ms-iot.vscode-ros
code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
code --install-extension twxs.cmake
```

### Additional Python Packages for Robotics

```bash
pip3 install -U \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    pyquaternion \
    transforms3d \
    open3d \
    trimesh \
    shapely
```

## Setting Up a ROS 2 Workspace

### Create and Initialize Workspace

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build --symlink-install
```

### Setup Workspace Environment

Add workspace to your shell profile:

```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Installing Gazebo Simulation Environment

### Install Gazebo Garden (Recommended)

```bash
# Add Gazebo's Ubuntu package repository
sudo apt install software-properties-common lsb-release
sudo add-apt-repository "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main"
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update

# Install Gazebo Garden
sudo apt install gazebo-garden
```

### Alternative: Install Gazebo Classic

```bash
sudo apt install ros-humble-gazebo-classic
```

## Installing NVIDIA Isaac Sim (Optional)

For advanced simulation with NVIDIA Isaac Sim:

### Prerequisites
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (version 470+)
- CUDA toolkit installed

### Installation Steps

```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535

# Reboot to apply driver changes
sudo reboot

# Verify GPU is detected
nvidia-smi

# Install Isaac Sim prerequisites
sudo apt install \
    python3-pip \
    python3-venv \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev
```

Note: Isaac Sim requires separate download and installation from NVIDIA's website.

## Verification and Testing

### Verify ROS 2 Installation

```bash
# Check ROS 2 version
printenv | grep -i ros

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Test basic ROS 2 functionality
ros2 --help
```

### Run Basic Tests

```bash
# Terminal 1: Start a publisher
source /opt/ros/humble/setup.bash
ros2 topic pub /chatter std_msgs/String "data: Hello ROS 2"

# Terminal 2: Start a subscriber
source /opt/ros/humble/setup.bash
ros2 topic echo /chatter
```

### Test Gazebo Installation

```bash
# Start Gazebo simulation
gazebo

# Or with ROS 2 integration
ros2 launch gazebo_ros empty_world.launch.py
```

## Common Configuration Files

### Create .bashrc Additions

Add these lines to your `~/.bashrc` for convenient ROS 2 usage:

```bash
# ROS 2 Humble Setup
source /opt/ros/humble/setup.bash

# ROS 2 Workspace (if created)
source ~/ros2_ws/install/setup.bash

# ROS 2 Environment Variables
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp

# Colcon completion
if [ -f /usr/share/colcon_cd/function/colcon_cd.sh ]; then
    source /usr/share/colcon_cd/function/colcon_cd.sh
    export _colcon_cd_root=~/ros2_ws
fi

# ROS 2 aliases
alias cb='cd ~/ros2_ws && colcon build --symlink-install'
alias cs='cd ~/ros2_ws/src'
alias cii='cd ~/ros2_ws && colcon build --packages-select'
```

### Create .vimrc for ROS 2 Development

```bash
cat << 'EOF' > ~/.vimrc
" Basic settings
set number
set tabstop=4
set shiftwidth=4
set expandtab
set autoindent
set smartindent
set wrap
set linebreak
set showmatch
set ignorecase
set smartcase

" Syntax highlighting
syntax on
set background=dark

" File type detection
filetype plugin indent on

" Python-specific settings
autocmd FileType python set tabstop=4
autocmd FileType python set shiftwidth=4
autocmd FileType python set expandtab
EOF
```

## Performance Optimization

### System Tuning for Real-time Performance

```bash
# Install real-time kernel (optional, for time-critical applications)
sudo apt install linux-image-rt-generic

# Set up CPU governor for performance
sudo apt install cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl disable ondemand
sudo systemctl enable cpufrequtils
```

### Network Configuration for Robotics

```bash
# Optimize network settings for ROS 2 communication
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.rmem_default=65536
sudo sysctl -w net.ipv4.tcp_rmem='4096 87380 134217728'
sudo sysctl -w net.ipv4.tcp_wmem='4096 65536 134217728'

# Make changes persistent
echo "net.core.rmem_max=134217728" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max=134217728" | sudo tee -a /etc/sysctl.conf
```

## Troubleshooting Common Issues

### ROS 2 Package Installation Issues

If you encounter issues with ROS 2 package installation:

```bash
# Clean package cache
sudo apt clean
sudo apt autoclean

# Update package lists
sudo apt update

# Fix broken packages
sudo apt --fix-broken install

# If still having issues, try manual installation:
sudo apt remove ros-humble-desktop
sudo apt autoremove
sudo apt update
sudo apt install ros-humble-desktop
```

### Python Package Issues

For Python package conflicts:

```bash
# Use virtual environments for ROS 2 projects
python3 -m venv ~/ros2_env
source ~/ros2_env/bin/activate
pip install -U pip setuptools
```

### Gazebo Not Starting

```bash
# Check if NVIDIA drivers are properly installed
nvidia-smi

# Install additional graphics libraries
sudo apt install \
    nvidia-prime \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri

# Test OpenGL
glxinfo | grep "OpenGL renderer"
```

## Development Best Practices

### Workspace Organization

```bash
# Recommended workspace structure
mkdir -p ~/ros2_ws/src
# - src/: Source code packages
# - build/: Build artifacts (auto-generated)
# - install/: Installation directory (auto-generated)
# - log/: Build logs (auto-generated)
```

### Version Control Setup

```bash
# Initialize git in your workspace
cd ~/ros2_ws
git init
echo "build/" >> .gitignore
echo "install/" >> .gitignore
echo "log/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
```

## Updating ROS 2

To keep your ROS 2 installation up to date:

```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update ROS 2 packages
sudo apt update
sudo apt upgrade ros-humble-*

# Update Python packages
pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U
```

## Uninstalling ROS 2

If you need to remove ROS 2:

```bash
# Remove ROS 2 packages
sudo apt remove ros-humble-*
sudo apt autoremove

# Remove repository
sudo rm /etc/apt/sources.list.d/ros2.list

# Remove GPG key
sudo rm /usr/share/keyrings/ros-archive-keyring.gpg

# Clean up environment (remove from ~/.bashrc)
# Manually remove the source command from ~/.bashrc
```

## Conclusion

You now have a complete Ubuntu 22.04 + ROS 2 Humble development environment ready for Physical AI and Robotics projects. The installation includes:

- Ubuntu 22.04 LTS with essential development tools
- ROS 2 Humble Hawksbill with desktop packages
- Gazebo simulation environment
- Development tools and configuration
- Performance optimizations for robotics applications

Your system is now ready for the 13-week Physical AI curriculum and beyond. Remember to regularly update your system and ROS 2 packages to maintain security and compatibility.