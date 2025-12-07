---
title: "Humanoid Kinematics"
sidebar_label: "Humanoid Kinematics"
description: "Understanding kinematics for humanoid robot motion and control"
---

# Humanoid Kinematics

## Introduction to Humanoid Kinematics

Humanoid kinematics is the study of motion in humanoid robots without considering the forces that cause the motion. It involves understanding how the joints and links of a humanoid robot move in relation to each other and how to compute positions, velocities, and orientations of various body parts.

## Types of Kinematics

### 1. Forward Kinematics

Forward kinematics calculates the position and orientation of the end effector (e.g., hand or foot) given the joint angles.

```python
import numpy as np
from math import sin, cos

class HumanoidForwardKinematics:
    def __init__(self):
        # Define link lengths for a simplified arm
        self.upper_arm_length = 0.3  # meters
        self.forearm_length = 0.25   # meters

    def dh_transform(self, a, alpha, d, theta):
        """Denavit-Hartenberg transformation matrix"""
        return np.array([
            [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def simple_arm_fk(self, shoulder_angle, elbow_angle):
        """Forward kinematics for a 2-DOF arm"""
        # Base transformation
        T01 = self.dh_transform(0, 0, 0, shoulder_angle)
        T12 = self.dh_transform(self.upper_arm_length, 0, 0, elbow_angle)
        T23 = self.dh_transform(self.forearm_length, 0, 0, 0)

        # Combined transformation
        T03 = T01 @ T12 @ T23

        # Extract end-effector position
        position = T03[:3, 3]
        orientation = T03[:3, :3]

        return position, orientation

    def full_body_fk(self, joint_angles):
        """Forward kinematics for full humanoid body"""
        # This is a simplified example - real implementations are more complex
        # joint_angles should contain all joint angles for the humanoid
        results = {}

        # Calculate positions for each limb
        for limb in ['left_arm', 'right_arm', 'left_leg', 'right_leg']:
            results[limb] = self.calculate_limb_fk(limb, joint_angles)

        return results

    def calculate_limb_fk(self, limb, joint_angles):
        """Calculate forward kinematics for a specific limb"""
        # Implementation would depend on the specific robot model
        # This is a placeholder
        pass
```

### 2. Inverse Kinematics

Inverse kinematics calculates the joint angles required to achieve a desired end-effector position and orientation.

```python
class HumanoidInverseKinematics:
    def __init__(self):
        self.upper_arm_length = 0.3
        self.forearm_length = 0.25

    def two_link_ik(self, target_x, target_y):
        """Inverse kinematics for a 2-DOF planar arm"""
        # Calculate distance from origin to target
        r = np.sqrt(target_x**2 + target_y**2)

        # Check if target is reachable
        if r > (self.upper_arm_length + self.forearm_length):
            # Target is out of reach - extend as far as possible
            print("Target out of reach, extending as far as possible")
            shoulder_angle = np.arctan2(target_y, target_x)
            elbow_angle = 0
        elif r < abs(self.upper_arm_length - self.forearm_length):
            # Target is inside reachable area
            print("Target is inside the reachable area")
            shoulder_angle = np.arctan2(target_y, target_x)
            elbow_angle = np.pi
        else:
            # Calculate elbow angle using law of cosines
            cos_elbow = (self.upper_arm_length**2 + self.forearm_length**2 - r**2) / (2 * self.upper_arm_length * self.forearm_length)
            elbow_angle = np.arccos(np.clip(cos_elbow, -1, 1))

            # Calculate shoulder angle
            k1 = self.upper_arm_length + self.forearm_length * cos(elbow_angle)
            k2 = self.forearm_length * sin(elbow_angle)
            shoulder_angle = np.arctan2(target_y, target_x) - np.arctan2(k2, k1)

        return shoulder_angle, elbow_angle

    def jacobian_ik(self, current_joints, target_position, end_effector_func, max_iterations=100, tolerance=1e-4):
        """Jacobian-based inverse kinematics using iterative method"""
        joints = current_joints.copy()

        for i in range(max_iterations):
            # Calculate current end-effector position
            current_pos, _ = end_effector_func(joints)

            # Calculate error
            error = target_position - current_pos

            if np.linalg.norm(error) < tolerance:
                break

            # Calculate Jacobian
            jacobian = self.calculate_jacobian(joints, end_effector_func)

            # Calculate joint updates using pseudo-inverse
            joint_delta = np.linalg.pinv(jacobian) @ error
            joints += joint_delta * 0.1  # Small step size for stability

        return joints

    def calculate_jacobian(self, joints, end_effector_func):
        """Calculate Jacobian matrix using numerical differentiation"""
        n_joints = len(joints)
        pos, _ = end_effector_func(joints)
        jacobian = np.zeros((3, n_joints))  # 3D position

        delta = 1e-6

        for i in range(n_joints):
            joints_plus = joints.copy()
            joints_minus = joints.copy()

            joints_plus[i] += delta
            joints_minus[i] -= delta

            pos_plus, _ = end_effector_func(joints_plus)
            pos_minus, _ = end_effector_func(joints_minus)

            jacobian[:, i] = (pos_plus - pos_minus) / (2 * delta)

        return jacobian
```

## Humanoid Robot Structure

### 1. Joint Types and Degrees of Freedom

Humanoid robots typically have the following joint configurations:

```python
class HumanoidStructure:
    def __init__(self):
        self.joint_definitions = {
            'head': {
                'joints': ['neck_yaw', 'neck_pitch', 'neck_roll'],
                'dof': 3,
                'range': [(-np.pi/3, np.pi/3), (-np.pi/4, np.pi/4), (-np.pi/6, np.pi/6)]
            },
            'left_arm': {
                'joints': ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
                          'left_elbow_pitch', 'left_wrist_yaw', 'left_wrist_pitch'],
                'dof': 6,
                'range': [(-np.pi, np.pi)] * 6
            },
            'right_arm': {
                'joints': ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
                          'right_elbow_pitch', 'right_wrist_yaw', 'right_wrist_pitch'],
                'dof': 6,
                'range': [(-np.pi, np.pi)] * 6
            },
            'left_leg': {
                'joints': ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
                          'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll'],
                'dof': 6,
                'range': [(-np.pi, np.pi)] * 6
            },
            'right_leg': {
                'joints': ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
                          'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll'],
                'dof': 6,
                'range': [(-np.pi, np.pi)] * 6
            }
        }

    def get_total_dof(self):
        """Calculate total degrees of freedom"""
        total_dof = 0
        for limb in self.joint_definitions.values():
            total_dof += limb['dof']
        return total_dof
```

### 2. Kinematic Chains

```python
class KinematicChain:
    def __init__(self, name, joint_names, link_lengths):
        self.name = name
        self.joint_names = joint_names
        self.link_lengths = link_lengths
        self.n_joints = len(joint_names)

    def get_pose(self, joint_angles):
        """Calculate the pose (position and orientation) of each link"""
        if len(joint_angles) != self.n_joints:
            raise ValueError("Number of joint angles must match number of joints")

        poses = []
        current_transform = np.eye(4)

        for i in range(self.n_joints):
            # Create transformation matrix for this joint
            joint_transform = self._joint_transform(joint_angles[i], self.link_lengths[i])
            current_transform = current_transform @ joint_transform
            poses.append(current_transform.copy())

        return poses

    def _joint_transform(self, angle, length):
        """Create transformation matrix for a single joint"""
        # This is a simplified example - actual implementation depends on joint type
        return np.array([
            [np.cos(angle), -np.sin(angle), 0, length * np.cos(angle)],
            [np.sin(angle), np.cos(angle), 0, length * np.sin(angle)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
```

## Balance and Center of Mass

### 1. Center of Mass Calculation

```python
class BalanceController:
    def __init__(self, robot_masses, robot_positions):
        """
        robot_masses: list of masses for each link
        robot_positions: list of positions for each link (x, y, z)
        """
        self.link_masses = robot_masses
        self.link_positions = robot_positions

    def calculate_com(self):
        """Calculate center of mass of the robot"""
        total_mass = sum(self.link_masses)
        com_x = sum(mass * pos[0] for mass, pos in zip(self.link_masses, self.link_positions)) / total_mass
        com_y = sum(mass * pos[1] for mass, pos in zip(self.link_masses, self.link_positions)) / total_mass
        com_z = sum(mass * pos[2] for mass, pos in zip(self.link_masses, self.link_positions)) / total_mass

        return np.array([com_x, com_y, com_z])

    def calculate_zmp(self, com_pos, com_vel, com_acc, gravity=9.81):
        """Calculate Zero Moment Point (ZMP)"""
        # ZMP = (x_com, y_com) - (h/g) * (x_com_ddot, y_com_ddot)
        # where h is height of COM above ground
        com_height = com_pos[2]  # z coordinate

        zmp_x = com_pos[0] - (com_height / gravity) * com_acc[0]
        zmp_y = com_pos[1] - (com_height / gravity) * com_acc[1]

        return np.array([zmp_x, zmp_y, 0])  # ZMP is on the ground plane
```

## Walking Patterns and Gait Generation

### 1. Inverse Kinematics for Walking

```python
class WalkingController:
    def __init__(self):
        self.step_length = 0.3  # meters
        self.step_height = 0.1  # meters
        self.step_duration = 1.0  # seconds

    def generate_walk_trajectory(self, num_steps, step_offset=0):
        """Generate foot trajectory for walking"""
        trajectory = []

        for step in range(num_steps):
            # Calculate step parameters
            start_time = step * self.step_duration
            end_time = (step + 1) * self.step_duration

            # Generate trajectory points
            for t in np.linspace(start_time, end_time, int(self.step_duration * 100)):
                # Calculate foot position based on phase of step
                phase = (t - start_time) / self.step_duration

                if phase < 0.5:  # Swing phase (foot moving forward)
                    x = step_offset + self.step_length * phase * 2
                    z = self.step_height * np.sin(phase * 2 * np.pi)  # Arc trajectory
                else:  # Stance phase (foot on ground)
                    x = step_offset + self.step_length
                    z = 0

                trajectory.append({
                    'time': t,
                    'position': np.array([x, 0, z]),
                    'phase': 'swing' if phase < 0.5 else 'stance'
                })

        return trajectory

    def calculate_leg_ik(self, foot_target, leg_type='left'):
        """Calculate inverse kinematics for leg to reach foot target"""
        # Simplified 3-DOF leg IK
        # foot_target: [x, y, z] position relative to hip
        x, y, z = foot_target

        # Calculate hip-to-foot distance
        d = np.sqrt(x**2 + y**2 + z**2)

        # Leg segment lengths (simplified model)
        thigh_length = 0.4  # meters
        shin_length = 0.4  # meters

        # Check reachability
        if d > (thigh_length + shin_length):
            # Scale down target to maximum reach
            scale = (thigh_length + shin_length) / d
            x *= scale
            y *= scale
            z *= scale
            d *= scale

        # Calculate hip joint angles
        hip_yaw = np.arctan2(y, x)
        hip_roll = 0  # Simplified
        hip_pitch = np.arctan2(z, np.sqrt(x**2 + y**2))

        # Calculate knee angle using law of cosines
        cos_knee = (thigh_length**2 + shin_length**2 - d**2) / (2 * thigh_length * shin_length)
        knee_pitch = np.pi - np.arccos(np.clip(cos_knee, -1, 1))

        # Calculate ankle angle
        ankle_pitch = hip_pitch - np.arctan2(z, np.sqrt(x**2 + y**2))

        if leg_type == 'right':
            # Right leg has inverted angles for some joints
            hip_roll = -hip_roll
            ankle_pitch = -ankle_pitch

        return {
            'hip_yaw': hip_yaw,
            'hip_roll': hip_roll,
            'hip_pitch': hip_pitch,
            'knee_pitch': knee_pitch,
            'ankle_pitch': ankle_pitch,
            'ankle_roll': 0  # Simplified
        }
```

## Control Strategies

### 1. PID Control for Joint Position

```python
class JointController:
    def __init__(self, kp=100, ki=1, kd=10, max_output=100):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.max_output = max_output

        self.prev_error = 0
        self.integral = 0

    def update(self, target_pos, current_pos, dt):
        """Update PID controller"""
        error = target_pos - current_pos

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Limit output
        output = np.clip(output, -self.max_output, self.max_output)

        self.prev_error = error

        return output

    def reset(self):
        """Reset integral and previous error"""
        self.integral = 0
        self.prev_error = 0
```

### 2. Trajectory Following

```python
class TrajectoryController:
    def __init__(self, trajectory_points, control_freq=100):
        self.trajectory = trajectory_points
        self.control_freq = control_freq
        self.current_index = 0

    def get_target_position(self, current_time):
        """Get target position for current time"""
        if self.current_index >= len(self.trajectory):
            return self.trajectory[-1]['position']

        # Find the closest trajectory point
        while (self.current_index < len(self.trajectory) - 1 and
               self.trajectory[self.current_index + 1]['time'] < current_time):
            self.current_index += 1

        if self.current_index == len(self.trajectory) - 1:
            return self.trajectory[self.current_index]['position']

        # Interpolate between current and next point
        current_point = self.trajectory[self.current_index]
        next_point = self.trajectory[self.current_index + 1]

        alpha = (current_time - current_point['time']) / (next_point['time'] - current_point['time'])
        target_pos = (1 - alpha) * current_point['position'] + alpha * next_point['position']

        return target_pos
```

## Humanoid Kinematics Challenges

### 1. Singularity Handling

```python
class SingularityHandler:
    def __init__(self):
        self.singularity_threshold = 0.001

    def is_singular(self, jacobian):
        """Check if Jacobian is near singular"""
        # Calculate determinant or condition number
        condition_number = np.linalg.cond(jacobian)
        return condition_number > 1.0 / self.singularity_threshold

    def damped_least_squares(self, jacobian, target_velocities, damping_factor=0.01):
        """Damped least squares method for singularity handling"""
        # J# = J^T * (J * J^T + λ^2 * I)^(-1)
        I = np.eye(jacobian.shape[0])
        damped_matrix = jacobian @ jacobian.T + damping_factor**2 * I
        jacobian_pseudo = jacobian.T @ np.linalg.inv(damped_matrix)

        joint_velocities = jacobian_pseudo @ target_velocities
        return joint_velocities
```

### 2. Redundancy Resolution

```python
class RedundancyResolver:
    def __init__(self, n_joints, n_cartesian_dof=6):
        self.n_joints = n_joints
        self.n_cartesian_dof = n_cartesian_dof

    def resolve_redundancy(self, jacobian, joint_angles, nullspace_target=None):
        """Resolve redundancy using nullspace projection"""
        if self.n_joints <= self.n_cartesian_dof:
            # Not redundant
            return np.linalg.pinv(jacobian)

        # Calculate nullspace
        # I - J# * J creates nullspace projection matrix
        jacobian_pseudo = np.linalg.pinv(jacobian)
        identity = np.eye(self.n_joints)
        nullspace_matrix = identity - jacobian_pseudo @ jacobian

        # Apply nullspace task if provided
        if nullspace_target is not None:
            joint_velocities = jacobian_pseudo @ target_velocities
            nullspace_velocities = nullspace_matrix @ nullspace_target
            return joint_velocities + nullspace_velocities
        else:
            return jacobian_pseudo
```

## Best Practices

### 1. Numerical Stability
- Use robust numerical methods for matrix inversion
- Implement singularity detection and handling
- Validate joint limits before applying commands
- Use appropriate coordinate systems (quaternions for rotations)

### 2. Computational Efficiency
- Pre-compute transformation matrices when possible
- Use analytical solutions where available
- Implement caching for repeated calculations
- Optimize for real-time performance

### 3. Safety Considerations
- Implement joint limit checking
- Monitor for excessive forces/torques
- Include emergency stop mechanisms
- Validate trajectories before execution

Humanoid kinematics is fundamental to creating natural and stable movement in humanoid robots. Understanding both forward and inverse kinematics, along with balance control and gait generation, enables the development of sophisticated humanoid behaviors.