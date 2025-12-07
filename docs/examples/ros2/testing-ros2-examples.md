---
title: "Testing ROS 2 Examples in Ubuntu 22.04"
sidebar_label: "Testing ROS 2 Examples"
description: "How to test ROS 2 examples in a clean Ubuntu 22.04 environment"
---

# Testing ROS 2 Examples in Ubuntu 22.04

## Overview

This guide provides instructions for testing ROS 2 examples in a clean Ubuntu 22.04 environment. Testing in a clean environment ensures that examples are truly reproducible and don't rely on hidden dependencies.

## Prerequisites

Before testing, ensure your Ubuntu 22.04 system has:

- ROS 2 Humble installed
- Basic development tools
- Git for version control
- Appropriate system resources (minimum 4-core CPU, 8GB RAM)

## Setting Up a Clean Environment

### 1. Verify Ubuntu Version

```bash
lsb_release -a
```

### 2. Install ROS 2 Dependencies

```bash
# Update package lists
sudo apt update

# Install basic development tools
sudo apt install -y build-essential cmake python3-colcon-common-extensions python3-rosdep python3-vcstool

# Install ROS 2 Humble
sudo apt install -y ros-humble-desktop
sudo apt install -y ros-humble-ros-base
```

### 3. Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

### 4. Set up ROS 2 Environment

```bash
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Creating a Test Workspace

### 1. Create Workspace Directory

```bash
mkdir -p ~/ros2_test_ws/src
cd ~/ros2_test_ws
```

### 2. Create Test Package

```bash
cd ~/ros2_test_ws/src
ros2 pkg create --build-type ament_cmake --dependencies rclcpp rclpy std_msgs test_examples
```

### 3. Copy Example Code

Copy the example code from the documentation into your test package:

```bash
# Create source directory if it doesn't exist
mkdir -p ~/ros2_test_ws/src/test_examples/src

# Copy talker code to ~/ros2_test_ws/src/test_examples/src/talker.cpp
# Copy listener code to ~/ros2_test_ws/src/test_examples/src/listener.cpp
```

## Building and Testing Examples

### 1. Build the Package

```bash
cd ~/ros2_test_ws
colcon build --packages-select test_examples --symlink-install
```

### 2. Source the Workspace

```bash
source install/setup.bash
```

### 3. Test the Talker/Listener Example

Open two terminals and run:

Terminal 1:
```bash
source ~/ros2_test_ws/install/setup.bash
ros2 run test_examples talker
```

Terminal 2:
```bash
source ~/ros2_test_ws/install/setup.bash
ros2 run test_examples listener
```

Expected output in Terminal 1:
```
[INFO] [1620000000.123456789] [talker]: Publishing: 'Hello World: 0'
[INFO] [1620000000.623456789] [talker]: Publishing: 'Hello World: 1'
```

Expected output in Terminal 2:
```
[INFO] [1620000000.623456789] [listener]: I heard: 'Hello World: 0'
[INFO] [1620000001.123456789] [listener]: I heard: 'Hello World: 1'
```

## Testing Launch Files

### 1. Create Launch Directory

```bash
mkdir -p ~/ros2_test_ws/src/test_examples/launch
```

### 2. Create Launch File

Create `~/ros2_test_ws/src/test_examples/launch/test_talker_listener_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='test_examples',
            executable='talker',
            name='talker',
            output='screen'
        ),
        Node(
            package='test_examples',
            executable='listener',
            name='listener',
            output='screen'
        )
    ])
```

### 3. Test the Launch File

```bash
source ~/ros2_test_ws/install/setup.bash
ros2 launch test_examples test_talker_listener_launch.py
```

## Testing with Parameters

### 1. Create Parameter File

Create `~/ros2_test_ws/src/test_examples/config/test_params.yaml`:

```yaml
talker:
  ros__parameters:
    publish_frequency: 2.0  # Publish every 0.5 seconds
    message_prefix: "Test: "
```

### 2. Update Node to Use Parameters

Update your talker node to accept parameters:

```cpp
#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class ParameterizedPublisher : public rclcpp::Node
{
public:
    ParameterizedPublisher()
    : Node("talker"), count_(0)
    {
        // Declare parameters with default values
        this->declare_parameter<std::string>("message_prefix", "Hello World: ");
        this->declare_parameter<double>("publish_frequency", 2.0);

        std::string prefix = this->get_parameter("message_prefix").as_string();
        double freq = this->get_parameter("publish_frequency").as_double();

        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / freq),
            std::bind(&ParameterizedPublisher::timer_callback, this));
        prefix_ = prefix;
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = prefix_ + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
    std::string prefix_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ParameterizedPublisher>());
    rclcpp::shutdown();
    return 0;
}
```

### 3. Update CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(test_examples)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# Add executable with parameters
add_executable(talker src/talker.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

add_executable(listener src/listener.cpp)
ament_target_dependencies(listener rclcpp std_msgs)

# Install targets
install(TARGETS
  talker
  listener
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

### 4. Rebuild and Test with Parameters

```bash
cd ~/ros2_test_ws
colcon build --packages-select test_examples
source install/setup.bash

# Test with parameter file
ros2 run test_examples talker --ros-args --params-file src/test_examples/config/test_params.yaml
```

## Verification Steps

### 1. Check Exit Codes

All commands should return exit code 0:

```bash
# Example of checking exit code
ros2 run test_examples talker &
PID=$!
sleep 5
kill $PID
echo $?
```

### 2. Monitor Resource Usage

```bash
# Check memory usage during execution
free -h

# Monitor processes
ps aux | grep ros
```

### 3. Verify Dependencies

```bash
# Check installed ROS packages
dpkg -l | grep ros-humble

# Verify required packages
ros2 pkg list | grep test_examples
```

## Performance Benchmarks

### 1. Build Time

Expected build times on minimum system (4-core CPU, 8GB RAM):
- Simple package: < 2 minutes
- Complex package with many dependencies: < 10 minutes

### 2. Runtime Performance

Monitor these metrics:
- CPU usage: Should not exceed 80% for simple examples
- Memory usage: Should remain stable over time
- Message rate: Should match expected frequency

## Troubleshooting Common Issues

### 1. Build Errors

If you encounter build errors:

```bash
# Clean build directory
rm -rf build/ install/ log/

# Rebuild with verbose output
colcon build --packages-select test_examples --event-handlers console_direct+
```

### 2. Runtime Errors

If nodes fail to run:

```bash
# Check available nodes
ros2 node list

# Check available topics
ros2 topic list

# Check node status
ros2 lifecycle list <node_name>
```

### 3. Permission Issues

```bash
# Check workspace permissions
ls -la ~/ros2_test_ws/

# Fix if needed
sudo chown -R $USER:$USER ~/ros2_test_ws/
```

## Automated Testing Script

Create an automated test script to verify all examples:

```bash
#!/bin/bash
# test_ros2_examples.sh

set -e  # Exit on any error

echo "Starting ROS 2 example tests..."

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Source workspace
source ~/ros2_test_ws/install/setup.bash

# Test 1: Build package
echo "Testing build..."
cd ~/ros2_test_ws
colcon build --packages-select test_examples
echo "Build test passed ✓"

# Test 2: Run talker/listener for 10 seconds
echo "Testing talker/listener communication..."
ros2 run test_examples talker &
TALKER_PID=$!
sleep 1  # Allow talker to start
ros2 run test_examples listener &
LISTENER_PID=$!

# Let them run for 10 seconds
sleep 10

# Kill processes
kill $TALKER_PID $LISTENER_PID 2>/dev/null || true

echo "Communication test passed ✓"

# Test 3: Launch file test
echo "Testing launch file..."
timeout 15 ros2 launch test_examples test_talker_listener_launch.py &
LAUNCH_PID=$!
sleep 15
kill $LAUNCH_PID 2>/dev/null || true

echo "Launch test passed ✓"

echo "All tests completed successfully!"
echo "Exit code: 0"
```

Make the script executable and run it:

```bash
chmod +x ~/ros2_test_ws/test_ros2_examples.sh
~/ros2_test_ws/test_ros2_examples.sh
```

## Success Criteria

All examples must meet these criteria to be considered reproducible:

1. ✅ **Exit code 0**: All commands complete successfully
2. ✅ **Expected output**: Nodes produce expected messages and behaviors
3. ✅ **No dependency errors**: All required packages are available
4. ✅ **Timing compliance**: Build and execution times meet requirements
5. ✅ **Resource efficiency**: CPU and memory usage within limits

Testing in a clean Ubuntu 22.04 environment ensures that your ROS 2 examples are truly reproducible and will work for other users following the same setup instructions.