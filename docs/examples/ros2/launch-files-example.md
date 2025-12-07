---
title: "ROS 2 Launch Files and Parameter Management"
sidebar_label: "Launch Files and Parameters"
description: "Learn to create launch files for managing multiple ROS 2 nodes and parameters"
---

# ROS 2 Launch Files and Parameter Management

## Introduction to Launch Files

Launch files in ROS 2 allow you to start multiple nodes with a single command, manage parameters, and handle complex robot configurations. They provide a convenient way to orchestrate your robotic system without manually starting each node.

## Launch File Syntax

ROS 2 uses Python for launch files. Here's the basic structure:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    return LaunchDescription([
        # Node definitions go here
    ])
```

## Simple Launch File Example

Create `~/ros2_ws/src/ros2_examples/launch/talker_listener_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros2_examples',
            executable='talker',
            name='talker',
            parameters=[
                {'param_name': 'param_value'}
            ],
            remappings=[
                ('original_topic', 'new_topic_name')
            ],
            output='screen'
        ),
        Node(
            package='ros2_examples',
            executable='listener',
            name='listener',
            output='screen'
        )
    ])
```

## Advanced Launch File with Parameters

Create `~/ros2_ws/src/ros2_examples/launch/robot_system_launch.py` with more complex configuration:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description':
                PathJoinSubstitution([
                    get_package_share_directory('ros2_examples'),
                    'urdf',
                    'my_robot.urdf'
                ])
            }
        ],
        output='screen'
    )

    # Joint state publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    # Navigation node
    navigation_node = Node(
        package='navigation_package',
        executable='navigation_node',
        name='navigation_node',
        parameters=[
            PathJoinSubstitution([
                get_package_share_directory('ros2_examples'),
                'config',
                'navigation.yaml'
            ]),
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name}
        ],
        remappings=[
            ('/cmd_vel', 'diff_drive_controller/cmd_vel_unstamped'),
            ('/odom', 'diff_drive_controller/odom'),
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_name,
        robot_state_publisher,
        joint_state_publisher,
        navigation_node
    ])
```

## YAML Parameter Files

ROS 2 supports YAML parameter files for complex configurations. Create `~/ros2_ws/src/ros2_examples/config/navigation.yaml`:

```yaml
navigation_node:
  ros__parameters:
    # Planner parameters
    planner_frequency: 5.0
    max_planning_retries: 10

    # Robot parameters
    robot_radius: 0.3
    max_vel_x: 0.5
    min_vel_x: 0.1
    max_vel_theta: 1.0
    min_vel_theta: 0.1

    # Goal checker parameters
    xy_goal_tolerance: 0.25
    yaw_goal_tolerance: 0.25

    # Global planner
    global_frame: "map"
    robot_base_frame: "base_link"

    # Recovery behaviors
    conservative_reset_dist: 3.0
    shutdown_costmaps_on_shutdown: true

    # TF specific parameters
    use_tf_static: true
    transform_tolerance: 0.3
```

## Launch File Best Practices

### 1. Parameter Organization

Group related parameters logically:

```python
# Good: Group navigation parameters
navigation_params = [
    {'planner_frequency': 5.0},
    {'max_planning_retries': 10},
    {'robot_radius': 0.3}
]

# Then use in node definition
navigation_node = Node(
    package='navigation_package',
    executable='navigation_node',
    parameters=navigation_params
)
```

### 2. Conditional Launch

Launch different nodes based on arguments:

```python
from launch.conditions import IfCondition, UnlessCondition
from launch.actions import DeclareLaunchArgument

# Declare argument
sim_mode = LaunchConfiguration('sim_mode')
declare_sim_mode = DeclareLaunchArgument(
    'sim_mode',
    default_value='false',
    description='Enable simulation mode'
)

# Conditional nodes
gazebo_spawner = Node(
    package='gazebo_ros',
    executable='spawn_entity.py',
    arguments=['-topic', 'robot_description', '-entity', 'my_robot'],
    condition=IfCondition(sim_mode)
)

real_robot_driver = Node(
    package='real_robot_driver',
    executable='driver_node',
    condition=UnlessCondition(sim_mode)
)
```

### 3. Remapping Topics

Use remappings to connect nodes with different topic names:

```python
Node(
    package='controller_package',
    executable='controller_node',
    remappings=[
        ('/cmd_vel', '/diff_drive/cmd_vel'),
        ('/odom', '/diff_drive/odom'),
        ('/tf', '/tf'),
        ('/tf_static', '/tf_static')
    ]
)
```

## Running Launch Files

### Basic Launch

```bash
# Run a launch file
ros2 launch ros2_examples talker_listener_launch.py

# Run with arguments
ros2 launch ros2_examples robot_system_launch.py use_sim_time:=true robot_name:=my_robot
```

### Debugging Launch Files

```bash
# Enable debug output
ros2 launch --debug ros2_examples talker_listener_launch.py

# Dry run to see what would be launched
ros2 launch --dry-run ros2_examples talker_listener_launch.py
```

## Multi-Node Launch Scenarios

### Robot Bringup

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Include other launch files
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nav2_bringup'),
            '/launch/navigation_launch.py'
        ]),
        launch_arguments={
            'use_sim_time': 'false'
        }.items()
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('slam_toolbox'),
            '/launch/online_async_launch.py'
        ])
    )

    return LaunchDescription([
        navigation_launch,
        slam_launch
    ])
```

## Parameter Management Strategies

### 1. Hierarchical Parameter Files

Organize parameters in multiple files for different subsystems:

```
config/
├── base.yaml          # Base robot parameters
├── navigation.yaml    # Navigation-specific parameters
├── sensors.yaml       # Sensor parameters
└── controllers.yaml   # Controller parameters
```

### 2. Runtime Parameter Updates

Use parameter services to update parameters at runtime:

```bash
# List parameters of a node
ros2 param list /navigation_node

# Get a specific parameter
ros2 param get /navigation_node planner_frequency

# Set a parameter
ros2 param set /navigation_node planner_frequency 10.0

# Load parameters from file
ros2 param load /navigation_node /path/to/params.yaml
```

## Common Launch File Patterns

### 1. Robot Description Loading

```python
from launch.substitutions import Command

# Load robot description from URDF
robot_description = Command([
    'xacro ',
    PathJoinSubstitution([
        get_package_share_directory('my_robot_description'),
        'urdf',
        'robot.xacro'
    ])
])

robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'robot_description': robot_description}]
)
```

### 2. Conditional Node Startup

```python
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

debug_mode = LaunchConfiguration('debug')
debug_node = Node(
    package='debug_package',
    executable='debug_node',
    condition=IfCondition(debug_mode)
)
```

## Troubleshooting Launch Files

### Common Issues

1. **Node not found**: Check package name and executable name
2. **Parameter errors**: Verify parameter names and types
3. **Path issues**: Use `get_package_share_directory` for package-relative paths
4. **Permission errors**: Ensure launch files are executable

### Useful Commands

```bash
# Check launch file syntax
python3 path/to/launch_file.py

# List all available launch files
ros2 launch -s

# Get help for launch arguments
ros2 launch ros2_examples robot_system_launch.py --show-args
```

Launch files are essential for managing complex robotic systems. They allow you to define complete robot configurations in a single file, making it easy to reproduce and share your robot setups.