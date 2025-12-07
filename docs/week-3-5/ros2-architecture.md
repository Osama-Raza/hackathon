---
title: "ROS 2 Architecture"
sidebar_label: "ROS 2 Architecture"
description: "Understanding ROS 2 core concepts, architecture, and differences from ROS 1"
---

# ROS 2 Architecture

## Introduction to ROS 2

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. Unlike traditional operating systems, ROS 2 provides services designed for a heterogeneous computer cluster such as hardware abstraction, device drivers, implementation of commonly used functionality, message-passing between processes, and package management.

## Core Concepts

### Nodes

Nodes are processes that perform computation. In ROS 2, nodes are lightweight and can be distributed across multiple machines. Each node is designed to perform a specific task and can communicate with other nodes through topics, services, and actions.

Key characteristics of nodes:
- Each node is a single process
- Nodes can be written in different programming languages (C++, Python, etc.)
- Nodes communicate with each other through topics, services, and actions
- Nodes are managed by a node lifecycle system

### Topics and Messages

Topics are named buses over which nodes exchange messages. Messages are data structures that are published to or subscribed from topics. The communication is asynchronous and follows a publish-subscribe pattern.

Example:
```python
# Publisher node
publisher = node.create_publisher(String, 'chatter', 10)
publisher.publish(String(data='Hello World'))

# Subscriber node
subscriber = node.create_subscription(String, 'chatter', callback_function, 10)
```

### Services

Services provide a request/reply communication pattern. When a node calls a service, it sends a request and waits for a response. This is synchronous communication and is useful for tasks that require a direct response.

### Actions

Actions are similar to services but are designed for long-running tasks. They provide feedback during execution and can be canceled. Actions are ideal for navigation, manipulation, or other tasks that take time to complete.

## Comparison with ROS 1

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Communication | Custom TCP/UDP | DDS-based |
| Middleware | Custom | DDS (Data Distribution Service) |
| Quality of Service | Limited | Extensive QoS controls |
| Real-time support | Limited | Improved real-time support |
| Multi-robot systems | Complex | Simplified |
| Security | Limited | Built-in security features |
| Cross-platform | Linux-focused | Cross-platform |

## DDS (Data Distribution Service)

ROS 2 uses DDS as its middleware. DDS provides:

- **Discovery**: Automatic discovery of nodes and topics
- **Quality of Service (QoS)**: Configurable communication policies
- **Reliability**: Guaranteed delivery options
- **Real-time performance**: Support for real-time applications
- **Security**: Built-in security features

### QoS Policies

Quality of Service policies allow fine-tuning of communication behavior:

- **Reliability**: Reliable vs. best-effort delivery
- **Durability**: Volatile vs. transient-local history
- **Deadline**: Time constraints for message delivery
- **Liveliness**: Mechanisms to detect if publishers are active

## Package Structure

ROS 2 packages follow a standard structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml            # Package metadata
├── src/                   # Source code
│   ├── node1.cpp
│   └── node2.cpp
├── include/               # Header files
├── launch/                # Launch files
├── config/                # Configuration files
├── test/                  # Test files
└── scripts/               # Python scripts
```

## When to Use ROS 2

ROS 2 is ideal for:

- **Multi-robot systems**: Better support for complex multi-robot deployments
- **Real-time applications**: Improved real-time capabilities
- **Production environments**: Enhanced stability and security
- **Cross-platform projects**: Better support for different operating systems
- **Commercial applications**: Better licensing and support options

## ROS 2 Distributions

ROS 2 follows a time-based release schedule with distributions named alphabetically:

- **Foxy Fitzroy** (2020) - LTS
- **Galactic Geochelone** (2021) - Short-term support
- **Humble Hawksbill** (2022) - LTS (Long Term Support)
- **Iron Irwini** (2023) - Short-term support
- **Jazzy Jalisco** (2024) - Short-term support

**Humble Hawksbill** is the current LTS (Long Term Support) version, recommended for new projects requiring long-term stability.

## Architecture Components

### RMW (ROS Middleware) Layer

The ROS Middleware layer abstracts the underlying DDS implementation, allowing ROS 2 to work with different DDS vendors (e.g., Fast DDS, Cyclone DDS, RTI Connext).

### rcl and rclcpp/rclpy

- **rcl**: C client library that provides the core ROS 2 functionality
- **rclcpp**: C++ client library built on top of rcl
- **rclpy**: Python client library built on top of rcl

### ros2cli

Command-line tools for interacting with ROS 2 systems, including:
- `ros2 run`: Run nodes
- `ros2 topic`: Interact with topics
- `ros2 service`: Interact with services
- `ros2 action`: Interact with actions
- `ros2 launch`: Launch multiple nodes

## Best Practices

1. **Modular Design**: Keep nodes focused on single responsibilities
2. **Appropriate QoS**: Choose QoS settings based on your application requirements
3. **Lifecycle Management**: Use lifecycle nodes for complex initialization and cleanup
4. **Parameter Management**: Use parameters for runtime configuration
5. **Logging**: Implement proper logging for debugging and monitoring
6. **Testing**: Write unit and integration tests for your nodes

Understanding ROS 2 architecture is fundamental to developing effective robotic applications. The next sections will explore practical implementation of these concepts through examples and hands-on exercises.