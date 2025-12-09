---
title: "Troubleshooting Guide"
sidebar_label: "Troubleshooting"
description: "Comprehensive troubleshooting guide for Physical AI and Robotics development"
---

# Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered during Physical AI and Robotics development with ROS 2, Ubuntu 22.04, and related tools. Each section addresses specific problems with step-by-step solutions and preventive measures.

## ROS 2 Common Issues

### 1. ROS 2 Environment Not Sourced

**Problem**: Commands like `ros2` or `colcon` are not recognized.

**Symptoms**:
- `Command 'ros2' not found`
- `Command 'colcon' not found`
- `ROS_DISTRO` environment variable not set

**Solutions**:
```bash
# Temporary fix - source the environment
source /opt/ros/humble/setup.bash

# Permanent fix - add to shell profile
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

**Prevention**: Always ensure the ROS 2 environment is sourced in your shell profile.

### 2. Package Installation Failures

**Problem**: ROS 2 packages fail to install with dependency errors.

**Symptoms**:
- `E: Unable to locate package ros-humble-package-name`
- Dependency conflicts during installation
- Broken package dependencies

**Solutions**:
```bash
# Update package lists
sudo apt update

# Fix broken packages
sudo apt --fix-broken install

# Clean package cache
sudo apt clean
sudo apt autoclean

# Re-add ROS 2 repository if needed
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

# Try installation again
sudo apt install ros-humble-package-name
```

### 3. ROS 2 Domain ID Conflicts

**Problem**: Nodes from different machines or processes cannot communicate.

**Symptoms**:
- Nodes cannot see each other
- Topics/services are not discovered
- Communication issues between processes

**Solutions**:
```bash
# Check current domain ID
echo $ROS_DOMAIN_ID

# Set domain ID explicitly
export ROS_DOMAIN_ID=0

# Or set in shell profile
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc

# For temporary use in a terminal
ROS_DOMAIN_ID=0 ros2 run package_name executable_name
```

### 4. RMW Implementation Issues

**Problem**: Different middleware implementations causing communication issues.

**Symptoms**:
- Inconsistent message delivery
- Performance issues
- Communication failures

**Solutions**:
```bash
# Check available RMW implementations
ls /opt/ros/humble/lib/python3.10/site-packages/rmw_*

# Set RMW implementation
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp

# Or in shell profile
echo "export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp" >> ~/.bashrc
```

## Build System Issues

### 1. Colcon Build Failures

**Problem**: `colcon build` fails with compilation errors.

**Symptoms**:
- CMake configuration errors
- Missing dependencies
- Linking errors
- Compiler errors

**Solutions**:
```bash
# Clean build directory
rm -rf ~/ros2_ws/build ~/ros2_ws/install ~/ros2_ws/log

# Install missing dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build with specific package
colcon build --packages-select package_name

# Build with more verbose output
colcon build --event-handlers console_direct+

# Build with symlinks to save space
colcon build --symlink-install
```

### 2. Dependency Resolution Issues

**Problem**: Missing dependencies during build process.

**Symptoms**:
- `Could not find a package configuration file`
- `Package 'package_name' not found`
- Missing CMake modules

**Solutions**:
```bash
# Update rosdep database
rosdep update

# Install dependencies for workspace
rosdep install --from-paths ~/ros2_ws/src --ignore-src -r -y

# Install specific package dependencies
sudo apt update
sudo apt install ros-humble-package-name ros-humble-package-name-dev
```

## Gazebo Simulation Issues

### 1. Gazebo Not Starting

**Problem**: Gazebo fails to launch or crashes immediately.

**Symptoms**:
- Segmentation fault errors
- OpenGL/GLX errors
- GPU driver issues

**Solutions**:
```bash
# Check GPU information
lspci | grep -E "VGA|3D"
nvidia-smi  # For NVIDIA GPUs

# Install graphics libraries
sudo apt install \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    nvidia-prime

# Test OpenGL
glxinfo | grep "OpenGL renderer"

# Try software rendering if hardware acceleration fails
export LIBGL_ALWAYS_SOFTWARE=1
gazebo
```

### 2. Gazebo-ROS Integration Problems

**Problem**: ROS 2 nodes cannot communicate with Gazebo simulation.

**Symptoms**:
- Topics not available in Gazebo
- Robot models not responding to commands
- Sensor data not publishing

**Solutions**:
```bash
# Launch Gazebo with ROS 2 bridge
ros2 launch gazebo_ros empty_world.launch.py

# Check available topics
ros2 topic list

# Verify Gazebo plugins are loaded
# Check model files for proper plugin configuration
# Ensure plugins are correctly specified in URDF/SDF
```

## Python Environment Issues

### 1. Python Package Conflicts

**Problem**: ROS 2 Python packages conflict with system Python packages.

**Symptoms**:
- Import errors
- Version conflicts
- Package not found errors

**Solutions**:
```bash
# Use virtual environments for projects
python3 -m venv ~/ros2_env
source ~/ros2_env/bin/activate
pip install -U pip setuptools

# Or use system Python with ROS 2
# Ensure ROS 2 environment is sourced before Python usage
source /opt/ros/humble/setup.bash
python3 -c "import rclpy; print('ROS 2 Python packages working')"
```

### 2. OpenCV Installation Issues

**Problem**: OpenCV not working with ROS 2 or causing conflicts.

**Symptoms**:
- Import errors in Python
- Camera nodes not working
- Image processing failures

**Solutions**:
```bash
# Install OpenCV for Python
pip3 install opencv-python

# Or install system package
sudo apt install python3-opencv

# Check OpenCV installation
python3 -c "import cv2; print(cv2.__version__)"
```

## Network and Communication Issues

### 1. Network Discovery Problems

**Problem**: ROS 2 nodes cannot discover each other over network.

**Symptoms**:
- Nodes on different machines cannot communicate
- Topics/services not visible across machines
- Multi-robot communication failures

**Solutions**:
```bash
# Check network connectivity
ping other_machine_ip

# Verify ROS domain ID is the same on all machines
echo $ROS_DOMAIN_ID

# Check firewall settings
sudo ufw status
# If firewall is blocking, allow ROS 2 ports:
sudo ufw allow 11311:65535/tcp
sudo ufw allow 11311:65535/udp

# Use specific network interface if needed
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
```

### 2. Topic Communication Issues

**Problem**: Topics are not publishing/subscribing correctly.

**Symptoms**:
- No data on topics
- High latency in communication
- Intermittent message delivery

**Solutions**:
```bash
# Check topic status
ros2 topic list
ros2 topic info /topic_name

# Check quality of service settings
ros2 topic info /topic_name --verbose

# Test with simple publisher/subscriber
ros2 topic pub /test_topic std_msgs/String "data: test" -1
ros2 topic echo /test_topic

# Adjust QoS settings if needed
# Use reliable vs best effort based on requirements
```

## Hardware Interface Issues

### 1. Serial Communication Problems

**Problem**: Robot hardware cannot communicate via serial interface.

**Symptoms**:
- Serial port not accessible
- Permission denied errors
- Communication timeouts

**Solutions**:
```bash
# Check available serial ports
ls /dev/tty*

# Check permissions
ls -l /dev/ttyUSB0

# Add user to dialout group
sudo usermod -a -G dialout $USER
# Note: You need to log out and back in for changes to take effect

# Check serial port settings
stty -F /dev/ttyUSB0 -a

# Test serial communication
echo "test" > /dev/ttyUSB0
cat /dev/ttyUSB0
```

### 2. USB Device Recognition

**Problem**: USB devices (cameras, sensors) not recognized by system.

**Symptoms**:
- Device not appearing in /dev
- Camera nodes fail to initialize
- Sensor data not available

**Solutions**:
```bash
# Check USB devices
lsusb

# Check device permissions
ls -l /dev/bus/usb/*

# Add udev rules for persistent permissions
# Create /etc/udev/rules.d/99-robot-permissions.rules
sudo tee /etc/udev/rules.d/99-robot-permissions.rules << EOF
SUBSYSTEM=="usb", ATTRS{idVendor}=="*", ATTRS{idProduct}=="*", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="*", ATTRS{idProduct}=="*", MODE="0666"
EOF

# Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Performance Issues

### 1. High CPU Usage

**Problem**: ROS 2 nodes consuming excessive CPU resources.

**Symptoms**:
- High CPU usage shown in htop
- System slowdown during robot operation
- Real-time performance issues

**Solutions**:
```bash
# Monitor processes
htop

# Check specific ROS 2 nodes
ros2 run demo_nodes_cpp talker
# In another terminal: htop to see CPU usage

# Optimize node execution
# Use appropriate spin rates in nodes
# Consider multithreaded executors for heavy processing
# Use timers instead of busy loops

# System optimization
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf
echo 'kernel.sched_migration_cost_ns=5000000' | sudo tee -a /etc/sysctl.conf
```

### 2. Memory Leaks

**Problem**: ROS 2 processes consuming increasing amounts of memory.

**Symptoms**:
- Gradually increasing memory usage
- System running out of memory
- Nodes crashing due to memory issues

**Solutions**:
```bash
# Monitor memory usage
watch -n 1 free -h

# Check specific process memory
ps aux | grep ros2

# Use memory debugging tools
# Install valgrind for C++ debugging
sudo apt install valgrind

# For Python memory debugging
pip3 install memory-profiler
python3 -m memory_profiler your_script.py
```

## Simulation-Specific Issues

### 1. Physics Simulation Instability

**Problem**: Robot models behaving unrealistically in simulation.

**Symptoms**:
- Robot falling through ground
- Unstable joint movements
- Physics objects behaving erratically

**Solutions**:
```bash
# Check Gazebo physics parameters in world file
# Adjust parameters like:
# - max_step_size
# - real_time_factor
# - max_contacts

# Example physics configuration:
# <physics type="ode">
#   <max_step_size>0.001</max_step_size>
#   <real_time_factor>1</real_time_factor>
#   <max_contacts>10</max_contacts>
# </physics>

# Reduce simulation step size for stability
# In launch files, set smaller step sizes
```

### 2. Sensor Simulation Issues

**Problem**: Simulated sensors not providing realistic data.

**Symptoms**:
- No sensor data being published
- Unrealistic sensor readings
- Sensor topics not updating

**Solutions**:
```bash
# Check sensor plugin configuration in SDF/URDF
# Verify plugin is properly loaded
# Check topic names and message types

# Test sensor topics
ros2 topic list | grep sensor
ros2 topic echo /sensor_topic_name

# Verify Gazebo plugins are working
gz topic -l  # List available Gazebo topics
```

## Development Environment Issues

### 1. IDE Integration Problems

**Problem**: Code editors not recognizing ROS 2 packages or providing incorrect autocompletion.

**Symptoms**:
- Import errors in IDE
- No autocompletion for ROS 2 packages
- Incorrect syntax highlighting

**Solutions**:
```bash
# For VS Code with ROS extension
# Ensure ROS 2 environment is sourced in terminal
source /opt/ros/humble/setup.bash
code .

# Set Python interpreter
# In VS Code: Ctrl+Shift+P → Python: Select Interpreter
# Choose the one from ROS 2 environment

# For terminal-based editors
# Source ROS 2 environment before starting editor
source /opt/ros/humble/setup.bash
vim your_file.py
```

### 2. Workspace Build Issues

**Problem**: Workspace fails to build or has inconsistent states.

**Symptoms**:
- Build succeeds but executables don't work
- Mixed build artifacts from different branches
- CMake cache conflicts

**Solutions**:
```bash
# Clean workspace completely
cd ~/ros2_ws
rm -rf build/ install/ log/

# Or use colcon to clean
colcon clean --build --install --log

# Rebuild everything
colcon build --symlink-install

# Check for mixed architectures (if working with different systems)
find ~/ros2_ws -name "*.so" -exec file {} \;
```

## Common Error Messages and Solutions

### "No executable found"
**Cause**: Package not built or not in PATH
**Solution**:
```bash
cd ~/ros2_ws
colcon build --packages-select package_name
source install/setup.bash
```

### "Failed to contact master"
**Cause**: ROS master not running or network issues
**Solution**: Ensure ROS 2 environment is sourced and check network settings

### "Segmentation fault"
**Cause**: Memory access violation, often due to pointer issues
**Solution**: Use debugging tools like gdb or valgrind to identify the issue

### "Permission denied"
**Cause**: Insufficient permissions for files/devices
**Solution**: Check file permissions and user group membership

### "Could not import"
**Cause**: Python path issues or missing dependencies
**Solution**: Source ROS 2 environment and install missing packages

## Preventive Measures

### 1. Regular System Maintenance
```bash
# Update system regularly
sudo apt update && sudo apt upgrade

# Clean package cache
sudo apt autoremove && sudo apt autoclean

# Backup important configurations
tar -czf ros2_backup_$(date +%Y%m%d).tar.gz ~/.bashrc ~/.vimrc ~/ros2_ws/src/
```

### 2. Environment Consistency
```bash
# Use consistent environment setup
# Always source ROS 2 environment in the same way
# Use version control for configuration files
```

### 3. Testing Practices
```bash
# Test changes incrementally
# Use simulation before testing on real hardware
# Implement proper error handling in code
```

## Getting Help

### Online Resources
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS Answers](https://answers.ros.org/questions/)
- [Gazebo Community](https://community.gazebosim.org/)
- [Ubuntu Community](https://discourse.ubuntu.com/)

### Debugging Commands
```bash
# General system info
uname -a
lsb_release -a
free -h
df -h

# ROS 2 specific info
ros2 doctor
ros2 run demo_nodes_cpp talker
ros2 run demo_nodes_cpp listener
```

This troubleshooting guide covers the most common issues encountered in Physical AI and Robotics development. When facing new problems, start by checking the basic environment setup, then work through the relevant section of this guide. Always ensure your system is up to date and that the ROS 2 environment is properly sourced before starting any development work.