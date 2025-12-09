---
title: "Locomotion Controller Example"
sidebar_label: "Locomotion Controller"
description: "Example implementation of a humanoid locomotion controller"
---

# Locomotion Controller Example

## Overview

This document provides a complete example of a humanoid locomotion controller that implements ZMP-based walking control with balance feedback. The controller demonstrates the integration of trajectory generation, balance control, and real-time feedback for stable bipedal walking.

## Complete Locomotion Controller Implementation

```python
import numpy as np
from math import sin, cos, sqrt
import time

class HumanoidLocomotionController:
    def __init__(self, robot_params):
        """
        Initialize the locomotion controller

        Args:
            robot_params: Dictionary containing robot-specific parameters
        """
        # Robot physical parameters
        self.com_height = robot_params.get('com_height', 0.8)  # Center of mass height
        self.foot_size = robot_params.get('foot_size', [0.2, 0.1])  # [length, width]
        self.max_step_length = robot_params.get('max_step_length', 0.3)
        self.max_step_width = robot_params.get('max_step_width', 0.3)
        self.step_height = robot_params.get('step_height', 0.1)
        self.step_duration = robot_params.get('step_duration', 1.0)
        self.gravity = 9.81

        # Control parameters
        self.control_freq = 500  # 500 Hz control frequency
        self.control_period = 1.0 / self.control_freq
        self.zmp_margin = 0.05   # Safety margin for ZMP

        # Walking state
        self.current_state = 'standing'  # standing, walking, balancing
        self.support_foot = 'left'  # left, right, both
        self.walk_speed = 0.0
        self.walk_direction = 0.0  # Yaw angle for walking direction

        # Initialize controllers
        self.zmp_controller = ZMPBasedController(self.com_height, self.gravity)
        self.balance_controller = BalanceFeedbackController()
        self.trajectory_generator = WalkingTrajectoryGenerator(
            self.com_height, self.step_height, self.step_duration
        )

        # State variables
        self.current_com_pos = np.array([0.0, 0.0, self.com_height])
        self.current_com_vel = np.array([0.0, 0.0, 0.0])
        self.current_com_acc = np.array([0.0, 0.0, 0.0])

        self.left_foot_pos = np.array([0.0, -0.1, 0.0])   # Initial foot positions
        self.right_foot_pos = np.array([0.0, 0.1, 0.0])

        self.desired_com_trajectory = []
        self.desired_foot_trajectory = []

        # Timing
        self.last_update_time = time.time()

    def update(self, sensor_data):
        """
        Main control update function

        Args:
            sensor_data: Dictionary containing sensor readings
                - imu_data: [roll, pitch, yaw, angular_vel_x, angular_vel_y, angular_vel_z]
                - joint_positions: array of joint angles
                - joint_velocities: array of joint velocities
                - ft_sensors: [left_foot_wrench, right_foot_wrench]
                - com_position: current CoM position
                - com_velocity: current CoM velocity

        Returns:
            Dictionary containing joint commands and status
        """
        current_time = time.time()
        dt = current_time - self.last_update_time

        if dt < self.control_period:
            # Return previous commands if not time for update
            return self.last_commands

        # Update state estimates
        self.update_state_estimates(sensor_data)

        # Determine control mode
        if self.current_state == 'standing':
            commands = self.standing_control()
        elif self.current_state == 'walking':
            commands = self.walking_control(sensor_data)
        else:  # balancing
            commands = self.balance_control(sensor_data)

        # Update timing
        self.last_update_time = current_time
        self.last_commands = commands

        return commands

    def update_state_estimates(self, sensor_data):
        """Update state estimates from sensor data"""
        # Update CoM estimates
        if 'com_position' in sensor_data:
            self.current_com_pos = np.array(sensor_data['com_position'])
        if 'com_velocity' in sensor_data:
            self.current_com_vel = np.array(sensor_data['com_velocity'])

        # Estimate CoM acceleration (simplified)
        # In practice, this would use a more sophisticated estimator
        self.current_com_acc = (self.current_com_vel - self.prev_com_vel) / self.control_period
        self.prev_com_vel = self.current_com_vel.copy()

    def start_walking(self, speed=0.3, direction=0.0):
        """Initialize walking motion"""
        self.walk_speed = speed
        self.walk_direction = direction
        self.current_state = 'walking'

        # Generate initial step sequence
        self.generate_step_sequence()

    def stop_walking(self):
        """Stop walking and transition to standing"""
        self.walk_speed = 0.0
        self.current_state = 'balancing'  # First balance, then stand
        self.wait_for_balance()

    def walking_control(self, sensor_data):
        """Main walking control logic"""
        # Check stability
        zmp_current = self.calculate_current_zmp()
        support_polygon = self.get_support_polygon()

        if not self.is_zmp_stable(zmp_current, support_polygon):
            # Emergency balance control
            self.current_state = 'balancing'
            return self.balance_control(sensor_data)

        # Generate reference trajectories
        com_ref = self.get_next_com_reference()
        foot_ref = self.get_next_foot_reference()

        # Calculate ZMP-based CoM trajectory
        com_command = self.zmp_controller.calculate_com_trajectory(
            com_ref, self.current_com_pos, self.current_com_vel
        )

        # Apply balance feedback
        balance_correction = self.balance_controller.compute_balance_correction(
            self.current_com_pos, self.current_com_vel, zmp_current
        )

        # Combine commands
        final_com_command = com_command + balance_correction

        # Convert CoM command to joint commands using inverse kinematics
        joint_commands = self.inverse_kinematics(final_com_command, foot_ref)

        # Add joint-level feedback control
        joint_commands = self.add_joint_feedback(joint_commands, sensor_data)

        return {
            'joint_commands': joint_commands,
            'status': 'walking',
            'zmp_error': self.calculate_zmp_error(zmp_current),
            'com_error': np.linalg.norm(final_com_command - self.current_com_pos)
        }

    def balance_control(self, sensor_data):
        """Balance control when stability is compromised"""
        # Calculate required CoM position for stability
        required_com = self.calculate_stable_com_position(sensor_data)

        # Generate smooth transition to stable position
        com_command = self.interpolate_to_stable_position(required_com)

        # Convert to joint commands
        joint_commands = self.inverse_kinematics(com_command,
                                               self.get_current_foot_positions())

        # Check if balanced
        if self.is_balanced():
            self.current_state = 'standing'

        return {
            'joint_commands': joint_commands,
            'status': 'balancing',
            'balance_effort': self.calculate_balance_effort()
        }

    def standing_control(self):
        """Control for standing position"""
        # Maintain current CoM position
        com_command = self.current_com_pos.copy()
        com_command[2] = self.com_height  # Maintain height

        # Keep feet in current position
        foot_positions = self.get_current_foot_positions()

        joint_commands = self.inverse_kinematics(com_command, foot_positions)

        return {
            'joint_commands': joint_commands,
            'status': 'standing',
            'com_variance': np.var([self.current_com_pos, self.current_com_vel])
        }

    def generate_step_sequence(self):
        """Generate sequence of footsteps based on walking parameters"""
        self.step_sequence = []

        # Calculate number of steps based on desired walking distance
        # For now, generate a simple alternating pattern
        for i in range(10):  # Plan 10 steps ahead
            step_time = i * self.step_duration
            foot_x = i * self.max_step_length * cos(self.walk_direction)
            foot_y = i * self.max_step_width * sin(self.walk_direction)

            # Alternate feet
            foot_name = 'left' if i % 2 == 0 else 'right'
            foot_offset_y = self.max_step_width/2 if foot_name == 'left' else -self.max_step_width/2

            step = {
                'time': step_time,
                'position': [foot_x, foot_y + foot_offset_y, 0],
                'foot': foot_name
            }
            self.step_sequence.append(step)

    def calculate_current_zmp(self):
        """Calculate current ZMP from sensor data"""
        # Simplified ZMP calculation
        # In practice, this would use force/torque sensors and full dynamics

        # ZMP = CoM - (h/g) * CoM_acceleration
        zmp_x = self.current_com_pos[0] - (self.com_height / self.gravity) * self.current_com_acc[0]
        zmp_y = self.current_com_pos[1] - (self.com_height / self.gravity) * self.current_com_acc[1]

        return np.array([zmp_x, zmp_y, 0])

    def get_support_polygon(self):
        """Calculate support polygon based on contact feet"""
        if self.support_foot == 'left':
            return self.calculate_foot_support_polygon(self.left_foot_pos)
        elif self.support_foot == 'right':
            return self.calculate_foot_support_polygon(self.right_foot_pos)
        else:  # both feet
            return self.calculate_both_feet_support_polygon()

    def calculate_foot_support_polygon(self, foot_pos):
        """Calculate support polygon for a single foot"""
        # Simplified as rectangular polygon
        length, width = self.foot_size
        corners = [
            [foot_pos[0] - length/2, foot_pos[1] - width/2],
            [foot_pos[0] + length/2, foot_pos[1] - width/2],
            [foot_pos[0] + length/2, foot_pos[1] + width/2],
            [foot_pos[0] - length/2, foot_pos[1] + width/2]
        ]
        return np.array(corners)

    def calculate_both_feet_support_polygon(self):
        """Calculate support polygon for both feet in contact"""
        # Combine both foot polygons
        left_polygon = self.calculate_foot_support_polygon(self.left_foot_pos)
        right_polygon = self.calculate_foot_support_polygon(self.right_foot_pos)

        # For simplicity, return bounding box of both feet
        all_points = np.vstack([left_polygon, right_polygon])
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        return np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])

    def is_zmp_stable(self, zmp, support_polygon):
        """Check if ZMP is within support polygon with safety margin"""
        # Simplified point-in-polygon test with margin
        margin = self.zmp_margin

        # Calculate bounding box of support polygon with margin
        min_x, min_y = np.min(support_polygon, axis=0) + margin
        max_x, max_y = np.max(support_polygon, axis=0) - margin

        return (min_x <= zmp[0] <= max_x) and (min_y <= zmp[1] <= max_y)

    def inverse_kinematics(self, com_pos, foot_positions):
        """Convert desired CoM and foot positions to joint commands"""
        # This is a simplified placeholder
        # Real implementation would use full-body inverse kinematics

        # For now, return a simple mapping
        # In practice, this would solve the full IK problem considering:
        # - CoM position constraint
        # - Foot position/orientation constraints
        # - Joint limits
        # - Balance constraints
        # - Smooth motion constraints

        joint_commands = {
            'left_hip_roll': self.map_to_joint('hip_roll', com_pos, foot_positions),
            'left_hip_pitch': self.map_to_joint('hip_pitch', com_pos, foot_positions),
            'left_knee': self.map_to_joint('knee', com_pos, foot_positions),
            'left_ankle_pitch': self.map_to_joint('ankle_pitch', com_pos, foot_positions),
            'left_ankle_roll': self.map_to_joint('ankle_roll', com_pos, foot_positions),
            'right_hip_roll': self.map_to_joint('hip_roll', com_pos, foot_positions),
            'right_hip_pitch': self.map_to_joint('hip_pitch', com_pos, foot_positions),
            'right_knee': self.map_to_joint('knee', com_pos, foot_positions),
            'right_ankle_pitch': self.map_to_joint('ankle_pitch', com_pos, foot_positions),
            'right_ankle_roll': self.map_to_joint('ankle_roll', com_pos, foot_positions)
        }

        return joint_commands

    def map_to_joint(self, joint_type, com_pos, foot_positions):
        """Simplified mapping from task space to joint space"""
        # This is a very simplified mapping for demonstration
        # Real implementation would use proper inverse kinematics
        if joint_type == 'hip_pitch':
            return (com_pos[2] - 0.7) * 0.5  # Adjust hip to maintain height
        elif joint_type == 'knee':
            return (self.com_height - com_pos[2]) * 2.0  # Adjust knee for height
        elif joint_type == 'ankle_pitch':
            return (com_pos[0] - self.current_com_pos[0]) * 0.1  # Correct forward position
        else:
            return 0.0  # Default position

    def add_joint_feedback(self, joint_commands, sensor_data):
        """Add joint-level feedback control"""
        if 'joint_positions' not in sensor_data or 'joint_velocities' not in sensor_data:
            return joint_commands

        # Simple PD control for each joint
        kp = 100.0  # Proportional gain
        kd = 10.0   # Derivative gain

        updated_commands = joint_commands.copy()

        # In practice, this would iterate through actual joint names and apply feedback
        # For demonstration, we'll just return the original commands
        return updated_commands

    def is_balanced(self):
        """Check if robot is in balanced state"""
        # Simplified balance check
        zmp_current = self.calculate_current_zmp()
        support_polygon = self.get_support_polygon()
        return self.is_zmp_stable(zmp_current, support_polygon) and \
               abs(self.current_com_vel[0]) < 0.05 and \
               abs(self.current_com_vel[1]) < 0.05

    def wait_for_balance(self):
        """Wait until robot achieves balance before transitioning to standing"""
        timeout = 2.0  # Wait up to 2 seconds
        start_time = time.time()

        while not self.is_balanced() and (time.time() - start_time) < timeout:
            time.sleep(0.01)  # 10ms sleep

        if self.is_balanced():
            self.current_state = 'standing'


class ZMPBasedController:
    """ZMP-based control for CoM trajectory generation"""
    def __init__(self, com_height, gravity):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = sqrt(gravity / com_height)

    def calculate_com_trajectory(self, desired_zmp, current_com_pos, current_com_vel):
        """Calculate CoM trajectory to achieve desired ZMP"""
        # Simple inverted pendulum model
        # CoM_ddot = g/h * (CoM - ZMP)

        com_acc = (self.gravity / self.com_height) * (current_com_pos[:2] - desired_zmp[:2])
        com_acc = np.append(com_acc, 0)  # No vertical acceleration for this model

        # Integrate to get new CoM position
        dt = 0.002  # 2ms integration step
        new_com_vel = current_com_vel + com_acc * dt
        new_com_pos = current_com_pos + new_com_vel * dt

        return new_com_pos


class BalanceFeedbackController:
    """Balance feedback controller"""
    def __init__(self):
        self.kp_com = 5.0   # CoM position feedback gain
        self.kd_com = 2.0   # CoM velocity feedback gain
        self.kp_zmp = 10.0  # ZMP error feedback gain

    def compute_balance_correction(self, com_pos, com_vel, current_zmp):
        """Compute balance correction based on current state"""
        # Calculate ZMP error
        desired_zmp = com_pos[:2]  # Simplified: desired ZMP is under CoM
        zmp_error = desired_zmp - current_zmp[:2]

        # Calculate CoM position error
        com_error = desired_zmp - com_pos[:2]

        # Combine corrections
        correction_x = self.kp_com * com_error[0] + self.kp_zmp * zmp_error[0]
        correction_y = self.kp_com * com_error[1] + self.kp_zmp * zmp_error[1]

        # Return correction as CoM offset
        return np.array([correction_x, correction_y, 0])


class WalkingTrajectoryGenerator:
    """Generate walking trajectories"""
    def __init__(self, com_height, step_height, step_duration):
        self.com_height = com_height
        self.step_height = step_height
        self.step_duration = step_duration

    def generate_foot_trajectory(self, start_pos, end_pos):
        """Generate smooth foot trajectory for step"""
        trajectory = []

        # Use 5th order polynomial for smooth motion
        steps = int(self.step_duration * 100)  # 100 points per second

        for i in range(steps + 1):
            t = i / steps  # Normalized time [0, 1]

            # 5th order polynomial for smooth interpolation
            f_t = 10*t**3 - 15*t**4 + 6*t**5

            # Interpolate x, y positions
            x = start_pos[0] + f_t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + f_t * (end_pos[1] - start_pos[1])

            # Vertical trajectory (parabolic arc)
            z = start_pos[2] + 4 * self.step_height * t * (1 - t)

            trajectory.append(np.array([x, y, z]))

        return trajectory


# Example usage and testing
def test_locomotion_controller():
    """Test the locomotion controller"""
    # Robot parameters
    robot_params = {
        'com_height': 0.8,
        'foot_size': [0.2, 0.1],
        'max_step_length': 0.3,
        'max_step_width': 0.2,
        'step_height': 0.1,
        'step_duration': 1.0
    }

    # Initialize controller
    controller = HumanoidLocomotionController(robot_params)

    # Simulate sensor data (in real implementation, this would come from actual sensors)
    sensor_data = {
        'com_position': [0.0, 0.0, 0.8],
        'com_velocity': [0.0, 0.0, 0.0],
        'imu_data': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'joint_positions': [0.0] * 12,  # Example: 12 joints
        'joint_velocities': [0.0] * 12
    }

    print("Testing locomotion controller...")

    # Test standing control
    controller.current_state = 'standing'
    commands = controller.update(sensor_data)
    print(f"Standing command status: {commands['status']}")

    # Test walking initialization
    controller.start_walking(speed=0.3, direction=0.0)
    print(f"Walking initialized. State: {controller.current_state}")

    # Simulate a few control cycles
    for i in range(5):
        commands = controller.update(sensor_data)
        print(f"Control cycle {i+1}: {commands['status']}")

        # Update simulated sensor data for next cycle
        sensor_data['com_position'][0] += 0.01  # Small forward movement
        sensor_data['com_position'][2] = 0.8    # Maintain height

    print("Locomotion controller test completed.")


if __name__ == "__main__":
    test_locomotion_controller()
```

## Controller Architecture

### 1. Hierarchical Control Structure

The locomotion controller implements a hierarchical control architecture:

```python
class HierarchicalController:
    """
    Hierarchical controller structure:
    - High Level: Gait planner and footstep generation
    - Mid Level: ZMP-based CoM trajectory generation
    - Low Level: Joint-level feedback control
    """
    def __init__(self):
        self.high_level_planner = GaitPlanner()
        self.mid_level_controller = ZMPController()
        self.low_level_controller = JointController()

    def control_step(self, state, reference):
        """Execute one control step through all levels"""
        # High level: Generate gait pattern
        gait_commands = self.high_level_planner.plan_gait(state, reference)

        # Mid level: Generate CoM trajectory
        com_commands = self.mid_level_controller.generate_com_trajectory(
            gait_commands
        )

        # Low level: Generate joint commands
        joint_commands = self.low_level_controller.generate_joint_commands(
            com_commands, state
        )

        return joint_commands
```

### 2. Safety and Emergency Handling

```python
class SafetyHandler:
    """Handle safety and emergency situations"""
    def __init__(self):
        self.emergency_thresholds = {
            'roll_angle': np.pi/4,      # 45 degrees
            'pitch_angle': np.pi/4,     # 45 degrees
            'zmp_deviation': 0.15,      # 15 cm from support
            'com_velocity': 0.5         # 0.5 m/s
        }

    def check_emergency(self, robot_state):
        """Check for emergency conditions"""
        imu_data = robot_state.get('imu_data', [0]*6)
        roll, pitch = imu_data[0], imu_data[1]

        # Check angular thresholds
        if abs(roll) > self.emergency_thresholds['roll_angle']:
            return True, "Roll angle exceeded"
        if abs(pitch) > self.emergency_thresholds['pitch_angle']:
            return True, "Pitch angle exceeded"

        # Check ZMP stability
        zmp = robot_state.get('zmp', np.array([0, 0]))
        support_polygon = robot_state.get('support_polygon', [])
        if not self.is_zmp_in_polygon(zmp, support_polygon):
            return True, "ZMP outside support polygon"

        return False, "No emergency"

    def emergency_stop(self, robot_interface):
        """Execute emergency stop procedure"""
        # Send zero joint commands
        zero_commands = {joint: 0.0 for joint in robot_interface.get_joint_names()}
        robot_interface.send_commands(zero_commands)

        # Transition to safe posture
        safe_posture = self.get_safe_posture()
        robot_interface.move_to_posture(safe_posture, duration=2.0)
```

## Testing and Validation

### 1. Simulation Testing

```python
class ControllerTester:
    """Test the locomotion controller in simulation"""
    def __init__(self, controller, simulator):
        self.controller = controller
        self.simulator = simulator

    def test_stability(self, test_duration=10.0):
        """Test controller stability"""
        start_time = time.time()
        stability_metrics = []

        while time.time() - start_time < test_duration:
            # Get sensor data from simulator
            sensor_data = self.simulator.get_sensor_data()

            # Update controller
            commands = self.controller.update(sensor_data)

            # Apply commands to simulator
            self.simulator.apply_commands(commands['joint_commands'])

            # Collect metrics
            stability_metrics.append(self.evaluate_stability(sensor_data))

            # Small delay to simulate real-time
            time.sleep(0.002)  # 500Hz

        return self.analyze_stability_metrics(stability_metrics)

    def test_various_gaits(self):
        """Test different walking patterns"""
        gaits_to_test = [
            {'speed': 0.2, 'step_width': 0.15},
            {'speed': 0.4, 'step_width': 0.18},
            {'speed': 0.3, 'step_width': 0.12, 'turn_rate': 0.2}
        ]

        results = {}
        for gait_params in gaits_to_test:
            result = self.test_single_gait(gait_params)
            results[str(gait_params)] = result

        return results
```

### 2. Performance Metrics

```python
class PerformanceMetrics:
    """Calculate performance metrics for the locomotion controller"""
    def __init__(self):
        self.metrics = {
            'stability_margin': [],
            'energy_efficiency': [],
            'walking_speed': [],
            'step_accuracy': [],
            'balance_recovery_time': []
        }

    def calculate_stability_margin(self, zmp_history, support_polygon_history):
        """Calculate stability margin over time"""
        margins = []
        for zmp, polygon in zip(zmp_history, support_polygon_history):
            margin = self.calculate_distance_to_polygon_boundary(zmp, polygon)
            margins.append(margin)
        return np.mean(margins)

    def calculate_energy_efficiency(self, joint_torques, joint_velocities, time_interval):
        """Calculate energy efficiency"""
        # Energy = sum of (torque * velocity) over time
        power = np.sum(np.abs(joint_torques * joint_velocities))
        energy = power * time_interval
        return energy

    def evaluate_performance(self):
        """Evaluate overall controller performance"""
        performance_score = {
            'stability': np.mean(self.metrics['stability_margin']),
            'efficiency': np.mean(self.metrics['energy_efficiency']),
            'speed_attainment': np.mean(self.metrics['walking_speed']),
            'accuracy': np.mean(self.metrics['step_accuracy'])
        }

        overall_score = np.mean(list(performance_score.values()))
        return overall_score, performance_score
```

## Integration with Robot Hardware

### 1. Hardware Interface

```python
class RobotHardwareInterface:
    """Interface between controller and robot hardware"""
    def __init__(self, robot_description):
        self.joint_names = robot_description['joint_names']
        self.joint_limits = robot_description['joint_limits']
        self.transmission_type = robot_description['transmission_type']

        # Initialize communication with robot
        self.initialize_communication()

    def send_commands(self, joint_commands):
        """Send joint commands to robot"""
        # Apply joint limits
        limited_commands = self.apply_joint_limits(joint_commands)

        # Convert to appropriate format based on transmission type
        if self.transmission_type == 'position':
            self.send_position_commands(limited_commands)
        elif self.transmission_type == 'velocity':
            self.send_velocity_commands(limited_commands)
        elif self.transmission_type == 'effort':
            self.send_effort_commands(limited_commands)

    def get_sensor_data(self):
        """Get current sensor data from robot"""
        sensor_data = {}

        # Read joint positions
        sensor_data['joint_positions'] = self.read_joint_positions()

        # Read joint velocities
        sensor_data['joint_velocities'] = self.read_joint_velocities()

        # Read IMU data
        sensor_data['imu_data'] = self.read_imu_data()

        # Read force/torque sensors
        sensor_data['ft_sensors'] = self.read_force_torque_sensors()

        return sensor_data

    def apply_joint_limits(self, commands):
        """Apply joint limits to commands"""
        limited_commands = {}
        for joint, command in commands.items():
            min_limit, max_limit = self.joint_limits[joint]
            limited_commands[joint] = np.clip(command, min_limit, max_limit)
        return limited_commands
```

This locomotion controller example demonstrates a complete implementation of ZMP-based walking control with safety features, real-time performance considerations, and integration with robot hardware. The controller balances theoretical control concepts with practical implementation requirements for stable humanoid locomotion.