---
title: "Building ROS 2 Packages"
sidebar_label: "Building ROS 2 Packages"
description: "Learn how to create and structure ROS 2 packages with CMakeLists.txt and package.xml"
---

# Building ROS 2 Packages

## Understanding ROS 2 Packages

A package is the basic building block of a ROS 2 system. It contains nodes, libraries, configuration files, and other resources needed for a specific functionality. Packages provide a way to organize and distribute ROS 2 code.

## Package Structure

A typical ROS 2 package has the following structure:

```
my_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package manifest
├── src/                    # Source code files
│   ├── publisher_node.cpp
│   └── subscriber_node.cpp
├── include/my_package/     # Header files
├── launch/                 # Launch files
├── config/                 # Configuration files
├── test/                   # Unit and integration tests
├── scripts/                # Python scripts
├── msg/                    # Custom message definitions
├── srv/                    # Custom service definitions
└── action/                 # Custom action definitions
```

## Creating a New Package

### Using colcon

The recommended way to create a new package is using the `ros2 pkg create` command:

```bash
ros2 pkg create --build-type ament_cmake my_robot_package
```

For Python packages:
```bash
ros2 pkg create --build-type ament_python my_robot_package
```

### Package Options

You can specify additional options when creating a package:

```bash
# Create with dependencies
ros2 pkg create --build-type ament_cmake --dependencies rclcpp rclpy std_msgs my_robot_package

# Create with nodes
ros2 pkg create --build-type ament_cmake --node-name my_node my_robot_package

# Create with custom maintainer info
ros2 pkg create --build-type ament_cmake --maintainer-email "user@example.com" --maintainer-name "Your Name" my_robot_package
```

## package.xml: The Package Manifest

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package for my robot</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

### Key Elements in package.xml

- **name**: The name of the package (must be unique in your workspace)
- **version**: Semantic versioning (MAJOR.MINOR.PATCH)
- **description**: Brief description of the package
- **maintainer**: Contact information for the package maintainer
- **license**: Software license information
- **buildtool_depend**: Build system used (ament_cmake, ament_python)
- **depend**: Runtime dependencies
- **build_depend**: Build-time dependencies
- **exec_depend**: Execution-time dependencies

## CMakeLists.txt: Build Configuration

For C++ packages, the `CMakeLists.txt` file defines how the package is built:

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# Add executable
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

# Install other files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Test configuration
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Package configuration
ament_package()
```

### Key CMake Directives

- **cmake_minimum_required**: Minimum CMake version required
- **project**: Define the project name
- **find_package**: Locate and load other packages
- **add_executable**: Create an executable target
- **ament_target_dependencies**: Specify dependencies for a target
- **install**: Specify files to install
- **ament_package**: Complete the package configuration

## Python Package Structure

For Python packages, the structure is slightly different:

```
my_python_package/
├── setup.py                # Python setup configuration
├── package.xml             # ROS 2 package manifest
├── my_python_package/      # Python module
│   ├── __init__.py
│   └── my_module.py
└── test/
```

The `setup.py` file for Python packages:

```python
from setuptools import setup

package_name = 'my_python_package'

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
    description='Example Python ROS 2 package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_python_package.my_node:main',
        ],
    },
)
```

## Building Your Package

### Using colcon

To build your package:

```bash
# Navigate to your workspace
cd ~/ros2_ws

# Build the specific package
colcon build --packages-select my_robot_package

# Build all packages (with symlinks for development)
colcon build --symlink-install

# Build with specific options
colcon build --packages-select my_robot_package --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Sourcing the Environment

After building, you need to source the setup files:

```bash
# Source the workspace
source install/setup.bash

# Or add to your .bashrc for persistent sourcing
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Common Build Issues and Solutions

### Missing Dependencies

If you encounter build errors due to missing dependencies:

```bash
# Install missing ROS packages
sudo apt update
sudo apt install ros-humble-<package-name>

# Or install multiple packages
sudo apt install ros-humble-rclcpp ros-humble-std-msgs ros-humble-sensor-msgs
```

### Permission Issues

If you encounter permission issues:

```bash
# Check permissions on your workspace
ls -la ~/ros2_ws/

# Ensure you own the files
sudo chown -R $USER:$USER ~/ros2_ws/
```

### Clean Build

If you encounter persistent build issues:

```bash
# Clean build artifacts
rm -rf build/ install/ log/

# Rebuild
colcon build --symlink-install
```

## Best Practices

1. **Descriptive Names**: Use clear, descriptive names for packages
2. **Single Responsibility**: Each package should have a single, well-defined purpose
3. **Dependency Management**: Only include necessary dependencies
4. **Version Control**: Use Git for version control with .gitignore files
5. **Documentation**: Include README files and inline documentation
6. **Testing**: Include unit tests for critical functionality
7. **Consistent Structure**: Follow standard ROS 2 package conventions

## Example: Complete Package Walkthrough

Let's create a simple publisher/subscriber example:

1. Create the package:
```bash
ros2 pkg create --build-type ament_cmake --dependencies rclcpp rclpy std_msgs example_talker
```

2. Add source files to the `src/` directory
3. Update `CMakeLists.txt` and `package.xml`
4. Build the package: `colcon build --packages-select example_talker`
5. Source the environment: `source install/setup.bash`
6. Run the nodes: `ros2 run example_talker talker` and `ros2 run example_talker listener`

Understanding how to properly structure and build ROS 2 packages is essential for developing modular, maintainable robotic applications. This foundation will help you create well-organized code that integrates seamlessly with the broader ROS 2 ecosystem.