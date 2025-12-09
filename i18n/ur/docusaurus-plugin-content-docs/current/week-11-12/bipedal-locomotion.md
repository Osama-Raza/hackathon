---
title: "Bipedal Locomotion Control"
sidebar_label: "Bipedal Locomotion Control"
description: "Control strategies for bipedal walking in humanoid robots"
---

# Bipedal Locomotion Control

## Introduction to Bipedal Locomotion

Bipedal locomotion is one of the most challenging aspects of humanoid robotics. Unlike wheeled robots or quadrupeds, bipedal robots must maintain balance on two legs while walking, which requires sophisticated control algorithms and precise timing. This chapter explores the fundamental principles and control strategies for achieving stable bipedal walking.

## Fundamentals of Bipedal Walking

### 1. Walking Phases

Bipedal walking consists of two main phases:

- **Stance Phase**: When the foot is in contact with the ground
- **Swing Phase**: When the foot is off the ground and moving forward

Additionally, there are two sub-phases:
- **Double Support Phase**: When both feet are on the ground (at the beginning and end of steps)
- **Single Support Phase**: When only one foot is on the ground

### 2. Key Parameters

```python
class WalkingParameters:
    def __init__(self):
        self.step_length = 0.3      # Distance between consecutive foot placements
        self.step_width = 0.2       # Distance between left and right foot centers
        self.step_height = 0.1      # Maximum height of swinging foot
        self.step_duration = 1.0    # Time to complete one step
        self.dsp_duration = 0.1     # Double support phase duration
        self.com_height = 0.8       # Desired center of mass height
        self.walk_speed = 0.3       # Average walking speed (m/s)
```

## Zero Moment Point (ZMP) Based Control

### 1. ZMP Theory

The Zero Moment Point (ZMP) is a critical concept in bipedal locomotion. It's the point on the ground where the moment of the ground reaction force equals zero.

```python
import numpy as np
from math import sqrt

class ZMPController:
    def __init__(self, robot_height=0.8, gravity=9.81):
        self.robot_height = robot_height
        self.gravity = gravity
        self.omega = sqrt(gravity / robot_height)  # Natural frequency

    def calculate_zmp(self, com_pos, com_acc):
        """
        Calculate ZMP from center of mass position and acceleration
        ZMP = (x_com, y_com) - (h/g) * (x_com_ddot, y_com_ddot)
        """
        zmp_x = com_pos[0] - (self.robot_height / self.gravity) * com_acc[0]
        zmp_y = com_pos[1] - (self.robot_height / self.gravity) * com_acc[1]
        return np.array([zmp_x, zmp_y])

    def com_trajectory_from_zmp(self, zmp_trajectory, initial_com, initial_vel):
        """
        Generate CoM trajectory from desired ZMP trajectory using the inverted pendulum model
        """
        com_trajectory = []
        com_pos = initial_com.copy()
        com_vel = initial_vel.copy()

        dt = 0.01  # Integration time step

        for zmp in zmp_trajectory:
            # Inverted pendulum dynamics: com_ddot = g/h * (com - zmp)
            com_acc = (self.gravity / self.robot_height) * (com_pos[:2] - zmp[:2])
            com_acc = np.append(com_acc, 0)  # No vertical acceleration for simplified model

            # Integrate to get velocity and position
            com_vel += com_acc * dt
            com_pos += com_vel * dt

            com_trajectory.append(com_pos.copy())

        return np.array(com_trajectory)
```

### 2. Preview Control

Preview control uses future ZMP references to generate stable CoM trajectories:

```python
class PreviewController:
    def __init__(self, zmp_ref, dt=0.01, preview_time=2.0):
        self.dt = dt
        self.preview_steps = int(preview_time / dt)
        self.zmp_ref = zmp_ref
        self.A = np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]])  # State transition
        self.B = np.array([dt**3/6, dt**2/2, dt])                      # Input matrix
        self.C = np.array([1, 0, -self.height])                       # Output matrix

    def calculate_control_gain(self, omega):
        """Calculate optimal control gains using discrete Riccati equation"""
        # This is a simplified version - full implementation requires solving Riccati equation
        Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]])  # State cost
        R = 1.0  # Control cost

        # Solve discrete algebraic Riccati equation (simplified)
        K = np.linalg.inv(self.B.T @ Q @ self.B + R) @ self.B.T @ Q @ self.A
        return K
```

## Walking Pattern Generation

### 1. Footstep Planning

```python
class FootstepPlanner:
    def __init__(self, step_length=0.3, step_width=0.2, max_turn_rate=np.pi/6):
        self.step_length = step_length
        self.step_width = step_width
        self.max_turn_rate = max_turn_rate

    def generate_footsteps(self, start_pos, goal_pos, start_yaw=0):
        """
        Generate a sequence of footsteps from start to goal position
        """
        footsteps = []

        # Calculate distance and direction
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        distance = sqrt(dx**2 + dy**2)
        target_yaw = np.arctan2(dy, dx)

        # Calculate number of steps needed
        n_steps = max(1, int(distance / self.step_length))

        # Generate footsteps
        current_pos = np.array(start_pos)
        current_yaw = start_yaw

        for i in range(n_steps + 1):
            # Calculate foot position (alternating left/right)
            if i % 2 == 0:  # Right foot
                foot_offset = np.array([-self.step_width/2 * np.sin(current_yaw),
                                       self.step_width/2 * np.cos(current_yaw)])
            else:  # Left foot
                foot_offset = np.array([self.step_width/2 * np.sin(current_yaw),
                                       -self.step_width/2 * np.cos(current_yaw)])

            foot_pos = current_pos + foot_offset

            footsteps.append({
                'position': foot_pos,
                'yaw': current_yaw,
                'step_type': 'right' if i % 2 == 0 else 'left',
                'time': i * 1.0  # 1 second per step
            })

            # Move to next step position
            step_vec = np.array([self.step_length * np.cos(current_yaw),
                                self.step_length * np.sin(current_yaw)])
            current_pos += step_vec

            # Adjust orientation if needed
            yaw_diff = target_yaw - current_yaw
            if abs(yaw_diff) > 0.1:  # If not aligned with goal
                current_yaw += np.clip(yaw_diff, -self.max_turn_rate * 0.5,
                                      self.max_turn_rate * 0.5)

        return footsteps
```

### 2. Trajectory Generation

```python
class WalkingTrajectoryGenerator:
    def __init__(self, step_height=0.1, com_height=0.8):
        self.step_height = step_height
        self.com_height = com_height

    def generate_swing_trajectory(self, start_pos, end_pos, step_duration=1.0):
        """Generate smooth trajectory for swinging foot"""
        trajectory = []

        # Use 5th order polynomial for smooth motion
        t_array = np.linspace(0, step_duration, int(step_duration * 100))

        # Calculate intermediate points
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        mid_z = max(start_pos[2], end_pos[2]) + self.step_height  # Peak height

        for t in t_array:
            # 5th order polynomial coefficients for smooth interpolation
            tau = t / step_duration
            f_tau = 10*tau**3 - 15*tau**4 + 6*tau**5  # Smooth interpolation function

            # Interpolate x, y positions
            x = start_pos[0] + f_tau * (end_pos[0] - start_pos[0])
            y = start_pos[1] + f_tau * (end_pos[1] - start_pos[1])

            # Vertical trajectory (parabolic arc)
            z_tau = 4 * self.step_height * tau * (1 - tau)  # Parabolic
            z = start_pos[2] + z_tau * (end_pos[2] - start_pos[2]) + z_tau * self.step_height

            trajectory.append(np.array([x, y, z]))

        return trajectory

    def generate_com_trajectory(self, footsteps, step_duration=1.0):
        """Generate CoM trajectory following ZMP stability criteria"""
        com_trajectory = []

        for i in range(len(footsteps) - 1):
            # Generate CoM trajectory between footsteps
            start_support = footsteps[i]['position']
            end_support = footsteps[i+1]['position']

            # Simple inverted pendulum model
            for t in np.linspace(0, step_duration, int(step_duration * 100)):
                # CoM should move towards the next support foot
                alpha = t / step_duration
                com_x = (1 - alpha) * start_support[0] + alpha * end_support[0]
                com_y = (1 - alpha) * start_support[1] + alpha * end_support[1]
                com_z = self.com_height  # Keep constant height

                com_trajectory.append(np.array([com_x, com_y, com_z]))

        return com_trajectory
```

## Balance Control

### 1. Feedback Control

```python
class BalanceController:
    def __init__(self, kp_com=10.0, kd_com=2.0, kp_ang=5.0, kd_ang=1.0):
        self.kp_com = kp_com  # Proportional gain for CoM position
        self.kd_com = kd_com  # Derivative gain for CoM velocity
        self.kp_ang = kp_ang  # Proportional gain for angle
        self.kd_ang = kd_ang  # Derivative gain for angular velocity

    def compute_balance_correction(self, current_state, desired_state):
        """
        current_state: {com_pos, com_vel, orientation, angular_vel}
        desired_state: {com_pos, orientation}
        """
        # CoM position error
        com_error = desired_state['com_pos'] - current_state['com_pos']
        com_vel_error = -current_state['com_vel']  # Assuming desired velocity is 0

        # Orientation error (simplified for pitch and roll)
        orientation_error = desired_state['orientation'] - current_state['orientation']
        angular_vel_error = -current_state['angular_vel']

        # Compute corrections
        com_correction = (self.kp_com * com_error +
                         self.kd_com * com_vel_error)
        angle_correction = (self.kp_ang * orientation_error +
                           self.kd_ang * angular_vel_error)

        return {
            'com_correction': com_correction,
            'angle_correction': angle_correction
        }
```

### 2. Capture Point Control

The Capture Point is the location where the robot must step to come to a complete stop.

```python
class CapturePointController:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.tau = sqrt(com_height / gravity)  # Time constant

    def calculate_capture_point(self, com_pos, com_vel):
        """Calculate the capture point"""
        capture_point = com_pos + self.tau * com_vel
        return capture_point

    def check_stability(self, capture_point, foot_position, foot_size=0.1):
        """Check if capture point is within support polygon"""
        # Simplified for a rectangular foot
        dx = abs(capture_point[0] - foot_position[0])
        dy = abs(capture_point[1] - foot_position[1])

        return dx <= foot_size/2 and dy <= foot_size/2
```

## Control Architecture

### 1. Hierarchical Control Structure

```python
class HierarchicalWalkingController:
    def __init__(self):
        self.footstep_planner = FootstepPlanner()
        self.trajectory_generator = WalkingTrajectoryGenerator()
        self.balance_controller = BalanceController()
        self.zmp_controller = ZMPController()

        # State variables
        self.current_support_foot = 'left'
        self.swing_trajectory = []
        self.com_trajectory = []

    def step(self, sensor_data, dt):
        """Main control step"""
        # 1. High-level: Plan footsteps if needed
        if self.should_plan_footsteps():
            self.plan_footsteps()

        # 2. Mid-level: Generate trajectories
        self.generate_trajectories()

        # 3. Low-level: Apply balance corrections
        balance_correction = self.balance_controller.compute_balance_correction(
            self.get_current_state(sensor_data),
            self.get_desired_state()
        )

        # 4. Generate joint commands
        joint_commands = self.compute_joint_commands(
            self.get_trajectory_reference(),
            balance_correction
        )

        return joint_commands

    def should_plan_footsteps(self):
        """Determine if new footsteps need to be planned"""
        # Check if we're approaching the end of current plan
        return len(self.remaining_footsteps) < 3

    def compute_joint_commands(self, trajectory_ref, balance_correction):
        """Convert desired trajectories to joint commands"""
        # This would typically involve inverse kinematics and joint-level control
        # Implementation depends on specific robot kinematics
        pass
```

## Walking Gait Patterns

### 1. Natural Walking Pattern

```python
class NaturalWalkingPattern:
    def __init__(self):
        self.joint_ranges = {
            'hip_pitch': (-0.5, 0.5),
            'knee_pitch': (0, 1.5),
            'ankle_pitch': (-0.5, 0.5),
            'ankle_roll': (-0.2, 0.2)
        }

    def generate_joint_trajectories(self, step_phase, step_progress):
        """
        step_phase: 'double_support', 'single_support'
        step_progress: 0 to 1 (progress through current step)
        """
        trajectories = {}

        if step_phase == 'double_support':
            # Both feet on ground, prepare for step
            trajectories['left_hip_pitch'] = self.smooth_interpolation(
                self.joint_ranges['hip_pitch'][0],
                self.joint_ranges['hip_pitch'][1] * 0.3,
                step_progress
            )
            trajectories['right_hip_pitch'] = self.smooth_interpolation(
                self.joint_ranges['hip_pitch'][0],
                self.joint_ranges['hip_pitch'][1] * 0.3,
                step_progress
            )
        else:  # single_support
            # One foot swinging, one supporting
            trajectories['left_hip_pitch'] = self.smooth_interpolation(
                self.joint_ranges['hip_pitch'][1] * 0.3,
                self.joint_ranges['hip_pitch'][0],
                step_progress
            )
            trajectories['right_hip_pitch'] = self.smooth_interpolation(
                self.joint_ranges['hip_pitch'][1] * 0.3,
                self.joint_ranges['hip_pitch'][1] * 0.7,
                step_progress
            )

        return trajectories

    def smooth_interpolation(self, start, end, progress):
        """Smooth interpolation using cosine function"""
        smooth_progress = 0.5 * (1 - np.cos(progress * np.pi))
        return start + (end - start) * smooth_progress
```

## Advanced Control Techniques

### 1. Model Predictive Control (MPC)

```python
class ModelPredictiveController:
    def __init__(self, prediction_horizon=10, control_horizon=3):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

    def solve_mpc(self, current_state, reference_trajectory):
        """
        Solve MPC optimization problem
        This is a simplified representation - real implementation would use
        quadratic programming solvers like CVXOPT or OSQP
        """
        # Define cost function: minimize tracking error and control effort
        # subject to system dynamics and constraints
        pass
```

### 2. Learning-Based Approaches

```python
import torch
import torch.nn as nn

class LearningBasedWalker(nn.Module):
    def __init__(self, state_dim=12, action_dim=6):
        super(LearningBasedWalker, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.policy(state)

    def get_action(self, state):
        """Get walking action from current state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.policy(state_tensor)
        return action.squeeze(0).detach().numpy()
```

## Stability Analysis

### 1. Stability Metrics

```python
class StabilityAnalyzer:
    def __init__(self):
        self.zmp_margin_threshold = 0.05  # 5cm safety margin
        self.roll_pitch_threshold = 0.2   # 0.2 radian threshold

    def evaluate_stability(self, robot_state):
        """Evaluate current stability of the robot"""
        metrics = {}

        # ZMP stability
        zmp = self.calculate_current_zmp(robot_state)
        support_polygon = self.get_support_polygon(robot_state)
        zmp_margin = self.calculate_zmp_margin(zmp, support_polygon)
        metrics['zmp_stable'] = zmp_margin > self.zmp_margin_threshold

        # Angular stability
        roll, pitch = robot_state['orientation'][:2]
        metrics['angular_stable'] = (abs(roll) < self.roll_pitch_threshold and
                                   abs(pitch) < self.roll_pitch_threshold)

        # CoM stability
        com_pos = robot_state['com_position']
        com_vel = robot_state['com_velocity']
        metrics['com_stable'] = self.is_com_stable(com_pos, com_vel)

        return metrics

    def calculate_zmp_margin(self, zmp, support_polygon):
        """Calculate minimum distance from ZMP to support polygon edge"""
        # Simplified calculation - real implementation would use geometric algorithms
        pass
```

## Implementation Considerations

### 1. Real-Time Performance

```python
class RealTimeWalkingController:
    def __init__(self, control_freq=500):  # 500 Hz control frequency
        self.control_period = 1.0 / control_freq
        self.last_update_time = 0

    def update(self, sensor_data):
        """Real-time control update with timing constraints"""
        import time

        current_time = time.time()
        elapsed = current_time - self.last_update_time

        if elapsed >= self.control_period:
            # Perform control calculations
            commands = self.compute_control_commands(sensor_data)

            # Update timing
            self.last_update_time = current_time

            return commands
        else:
            # Return previous commands if not time for update
            return self.last_commands
```

### 2. Safety Features

```python
class SafetyController:
    def __init__(self):
        self.emergency_stop_thresholds = {
            'roll': np.pi/3,      # 60 degrees
            'pitch': np.pi/3,     # 60 degrees
            'zmp_deviation': 0.1  # 10 cm from support
        }

    def check_safety(self, robot_state):
        """Check if robot is in safe operating conditions"""
        roll, pitch, _ = robot_state['orientation']

        if abs(roll) > self.emergency_stop_thresholds['roll']:
            return False, "Roll angle exceeded safety limit"

        if abs(pitch) > self.emergency_stop_thresholds['pitch']:
            return False, "Pitch angle exceeded safety limit"

        # Check ZMP position
        zmp = self.calculate_current_zmp(robot_state)
        support_polygon = self.get_support_polygon(robot_state)
        zmp_deviation = self.calculate_zmp_deviation(zmp, support_polygon)

        if zmp_deviation > self.emergency_stop_thresholds['zmp_deviation']:
            return False, "ZMP exceeded safety limit"

        return True, "Safe"
```

## Performance Evaluation

### 1. Walking Metrics

```python
class WalkingPerformanceEvaluator:
    def __init__(self):
        self.metrics = {
            'walking_speed': [],
            'energy_efficiency': [],
            'stability_margin': [],
            'step_accuracy': []
        }

    def evaluate_step(self, step_data):
        """Evaluate a single walking step"""
        # Calculate step length accuracy
        actual_step_length = np.linalg.norm(
            step_data['foot_end_pos'] - step_data['foot_start_pos']
        )
        error = abs(actual_step_length - step_data['desired_step_length'])

        # Calculate energy consumption
        energy = self.calculate_energy_consumption(step_data['joint_torques'])

        return {
            'step_accuracy': 1 - error / step_data['desired_step_length'],
            'energy_efficiency': energy,
            'stability': self.calculate_stability_margin(step_data)
        }
```

Bipedal locomotion control remains one of the most active research areas in humanoid robotics. Success requires careful integration of kinematics, dynamics, control theory, and real-time implementation considerations. The key to stable walking is maintaining the center of mass within the support polygon while generating natural, efficient movement patterns.