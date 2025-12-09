---
title: "ROS 2 Talker/Listener Example"
sidebar_label: "Talker/Listener Example"
description: "Complete example of ROS 2 publisher and subscriber nodes"
---

# ROS 2 Talker/Listener Example

## Overview

This example demonstrates the fundamental ROS 2 communication pattern using publisher and subscriber nodes. The talker node publishes messages to a topic, and the listener node subscribes to that topic to receive the messages.

## Prerequisites

Before running this example, ensure you have:

- Ubuntu 22.04 with ROS 2 Humble installed
- A properly configured ROS 2 workspace
- Basic understanding of ROS 2 concepts

## Creating the Package

First, create a new ROS 2 package for our example:

```bash
# Navigate to your ROS 2 workspace
cd ~/ros2_ws

# Create the package with dependencies
ros2 pkg create --build-type ament_cmake --dependencies rclcpp rclpy std_msgs ros2_examples
```

## C++ Talker Node

Create the talker node in `~/ros2_ws/src/ros2_examples/src/talker.cpp`:

```cpp
#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher()
    : Node("talker"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World: " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}
```

## C++ Listener Node

Create the listener node in `~/ros2_ws/src/ros2_examples/src/listener.cpp`:

```cpp
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber()
    : Node("listener")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
            });
    }

private:
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
```

## Package Configuration

Update the `package.xml` file in `~/ros2_ws/src/ros2_examples/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>ros2_examples</name>
  <version>0.0.0</version>
  <description>Examples for ROS 2 learning</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Update the `CMakeLists.txt` file in `~/ros2_ws/src/ros2_examples/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(ros2_examples)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# Add executables
add_executable(talker src/talker.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

add_executable(listener src/listener.cpp)
ament_target_dependencies(listener rclcpp std_msgs)

# Install executables
install(TARGETS
  talker
  listener
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## Building the Package

Build the package using colcon:

```bash
# Navigate to your workspace
cd ~/ros2_ws

# Build the specific package
colcon build --packages-select ros2_examples

# Source the workspace
source install/setup.bash
```

## Running the Example

Run the talker and listener nodes in separate terminals:

Terminal 1 (Talker):
```bash
source ~/ros2_ws/install/setup.bash
ros2 run ros2_examples talker
```

Terminal 2 (Listener):
```bash
source ~/ros2_ws/install/setup.bash
ros2 run ros2_examples listener
```

You should see output like:
```
[INFO] [1620000000.123456789] [talker]: Publishing: 'Hello World: 0'
[INFO] [1620000000.623456789] [talker]: Publishing: 'Hello World: 1'
[INFO] [1620000000.623456789] [listener]: I heard: 'Hello World: 0'
[INFO] [1620000001.123456789] [listener]: I heard: 'Hello World: 1'
```

## Python Alternative

You can also implement the same functionality in Python. Create `~/ros2_ws/src/ros2_examples/ros2_examples/talker.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Talker(Node):

    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    talker = Talker()
    rclpy.spin(talker)
    talker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

And create `~/ros2_ws/src/ros2_examples/ros2_examples/listener.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)
    listener = Listener()
    rclpy.spin(listener)
    listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

For Python packages, update the `setup.py`:

```python
from setuptools import setup

package_name = 'ros2_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Examples for ROS 2 learning',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = ros2_examples.talker:main',
            'listener = ros2_examples.listener:main',
        ],
    },
)
```

## Troubleshooting

### Common Issues

1. **Package not found**: Make sure you've sourced the workspace setup file
2. **Build errors**: Check that all dependencies are installed
3. **Permission errors**: Ensure you have write access to your workspace

### Useful Commands

- Check available nodes: `ros2 node list`
- Check available topics: `ros2 topic list`
- Echo a topic: `ros2 topic echo /topic std_msgs/msg/String`

This example demonstrates the fundamental publisher-subscriber pattern in ROS 2, which is essential for robot communication and coordination.