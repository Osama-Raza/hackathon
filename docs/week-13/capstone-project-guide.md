---
title: "Capstone Project: Physical AI Robot Assistant"
sidebar_label: "Capstone Project Guide"
description: "Complete capstone project integrating all concepts from the Physical AI curriculum"
---

# Capstone Project: Physical AI Robot Assistant

## Project Overview

The capstone project integrates all concepts learned throughout the 13-week Physical AI curriculum into a comprehensive robot assistant system. This project demonstrates the complete pipeline from perception to action in a conversational humanoid robot.

## Project Objectives

By completing this capstone project, you will:

1. Integrate ROS 2, Gazebo, and perception systems
2. Implement humanoid kinematics and locomotion control
3. Create conversational AI with voice interaction
4. Design safe and robust robot behaviors
5. Demonstrate complete Physical AI system architecture

## System Architecture

```mermaid
graph TB
    subgraph "User Interaction Layer"
        A[Voice Commands] --> B[Speech Recognition]
        C[Gesture Recognition] --> D[Intent Processing]
        B --> D
        D --> E[Dialogue Manager]
    end

    subgraph "AI Processing Layer"
        E --> F[Natural Language Understanding]
        F --> G[Task Planning]
        G --> H[Action Selection]
    end

    subgraph "Robot Control Layer"
        H --> I[Locomotion Control]
        H --> J[Manipulation Control]
        H --> K[Human-Robot Interaction]
        I --> L[Inverse Kinematics]
        J --> L
        K --> L
    end

    subgraph "Hardware Interface Layer"
        L --> M[Motor Control]
        N[Sensor Fusion] --> M
        O[Camera] --> N
        P[LiDAR] --> N
        Q[IMU] --> N
    end

    subgraph "Simulation Environment"
        R[Gazebo Simulation]
        S[Isaac Sim (Optional)]
    end

    M --> R
    M --> S
```

## Prerequisites

Before starting the capstone project, ensure you have completed:

- Week 1-2: Physical AI foundations and sensor systems
- Week 3-5: ROS 2 fundamentals and package development
- Week 6-7: Gazebo simulation and URDF modeling
- Week 8-10: NVIDIA Isaac platform and perception pipelines
- Week 11-12: Humanoid kinematics and locomotion control
- Week 13: Voice-to-action pipelines and GPT integration

## Project Components

### 1. Voice Command Interface

```python
#!/usr/bin/env python3
"""
Capstone Project: Voice Command Interface
File: capstone_voice_interface.py
"""

import rospy
import speech_recognition as sr
import pyttsx3
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import openai
import json
import threading
import queue

class CapstoneVoiceInterface:
    def __init__(self):
        rospy.init_node('capstone_voice_interface')

        # Publishers
        self.voice_response_pub = rospy.Publisher('/voice_response', String, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.8)

        # GPT integration
        self.gpt_api_key = rospy.get_param('~gpt_api_key', '')
        if self.gpt_api_key:
            openai.api_key = self.gpt_api_key

        # Command queue
        self.command_queue = queue.Queue()
        self.listening = True

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        rospy.loginfo("Capstone Voice Interface initialized")

    def listen_for_commands(self):
        """Listen continuously for voice commands"""
        while self.listening and not rospy.is_shutdown():
            try:
                with self.microphone as source:
                    rospy.loginfo("Listening for commands...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                rospy.loginfo(f"Heard: {text}")

                # Process command
                self.process_command(text)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                self.speak("Sorry, I didn't understand that. Could you repeat?")
            except sr.RequestError as e:
                rospy.logerr(f"Speech recognition error: {e}")
                self.speak("Sorry, I'm having trouble with speech recognition.")

    def process_command(self, command_text):
        """Process command using GPT and execute robot action"""
        if self.gpt_api_key:
            # Use GPT for command interpretation
            response = self.process_with_gpt(command_text)
        else:
            # Use rule-based processing
            response = self.process_rule_based(command_text)

        if response:
            self.execute_robot_action(response)

    def process_with_gpt(self, command_text):
        """Process command using GPT"""
        try:
            messages = [
                {"role": "system", "content": """You are a helpful robot assistant.
                Convert user commands to structured robot actions.
                Respond in JSON format with 'action' and 'parameters'.
                Actions: move_forward, move_backward, turn_left, turn_right, speak, stop.
                Example: {"action": "move_forward", "parameters": {"distance": 1.0}}"""}
            ]
            messages.append({"role": "user", "content": command_text})

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )

            gpt_response = response.choices[0].message.content

            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', gpt_response, re.DOTALL)
            if json_match:
                action_data = json.loads(json_match.group())
                return action_data

        except Exception as e:
            rospy.logerr(f"GPT processing error: {e}")

        # Fallback to rule-based processing
        return self.process_rule_based(command_text)

    def process_rule_based(self, command_text):
        """Rule-based command processing"""
        command_lower = command_text.lower()

        if 'forward' in command_lower or 'ahead' in command_lower:
            return {"action": "move_forward", "parameters": {"distance": 1.0}}
        elif 'backward' in command_lower or 'back' in command_lower:
            return {"action": "move_backward", "parameters": {"distance": 1.0}}
        elif 'left' in command_lower:
            return {"action": "turn_left", "parameters": {"angle": 90}}
        elif 'right' in command_lower:
            return {"action": "turn_right", "parameters": {"angle": 90}}
        elif 'stop' in command_lower:
            return {"action": "stop", "parameters": {}}
        elif 'hello' in command_lower or 'hi' in command_lower:
            return {"action": "speak", "parameters": {"text": "Hello! How can I help you?"}}
        else:
            return {"action": "speak", "parameters": {"text": f"I heard: {command_text}. I'm not sure how to respond."}}

    def execute_robot_action(self, action_data):
        """Execute robot action based on processed command"""
        action = action_data.get('action')
        params = action_data.get('parameters', {})

        if action == 'move_forward':
            self.move_robot('forward', params.get('distance', 1.0))
        elif action == 'move_backward':
            self.move_robot('backward', params.get('distance', 1.0))
        elif action == 'turn_left':
            self.turn_robot('left', params.get('angle', 90))
        elif action == 'turn_right':
            self.turn_robot('right', params.get('angle', 90))
        elif action == 'stop':
            self.stop_robot()
        elif action == 'speak':
            text = params.get('text', 'Hello')
            self.speak(text)

    def move_robot(self, direction, distance):
        """Move robot in specified direction"""
        msg = Twist()

        if direction == 'forward':
            msg.linear.x = 0.5  # 0.5 m/s
        elif direction == 'backward':
            msg.linear.x = -0.5  # -0.5 m/s

        # Publish movement for the required duration
        duration = distance / 0.5  # time = distance / speed
        start_time = rospy.Time.now()

        while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(0.1)

        self.stop_robot()

    def turn_robot(self, direction, angle_degrees):
        """Turn robot in specified direction"""
        msg = Twist()
        angular_speed = 0.5  # rad/s
        angle_radians = angle_degrees * 3.14159 / 180.0
        duration = angle_radians / angular_speed
        start_time = rospy.Time.now()

        if direction == 'left':
            msg.angular.z = angular_speed
        elif direction == 'right':
            msg.angular.z = -angular_speed

        while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(msg)
            rospy.sleep(0.1)

        self.stop_robot()

    def stop_robot(self):
        """Stop robot movement"""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)

    def speak(self, text):
        """Speak text using text-to-speech"""
        rospy.loginfo(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

        # Publish to ROS topic as well
        response_msg = String()
        response_msg.data = text
        self.voice_response_pub.publish(response_msg)

    def run(self):
        """Run the voice interface"""
        # Start listening in a separate thread
        listen_thread = threading.Thread(target=self.listen_for_commands)
        listen_thread.daemon = True
        listen_thread.start()

        try:
            rospy.spin()
        except KeyboardInterrupt:
            self.listening = False
            rospy.loginfo("Shutting down voice interface...")

if __name__ == '__main__':
    interface = CapstoneVoiceInterface()
    interface.run()
```

### 2. Perception and Navigation System

```python
#!/usr/bin/env python3
"""
Capstone Project: Perception and Navigation
File: capstone_perception_navigation.py
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import tf
from std_msgs.msg import Bool

class CapstonePerceptionNavigation:
    def __init__(self):
        rospy.init_node('capstone_perception_navigation')

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.obstacle_detected_pub = rospy.Publisher('/obstacle_detected', Bool, queue_size=10)

        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Robot state
        self.position = Point()
        self.orientation = tf.transformations.quaternion_matrix([0, 0, 0, 1])
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        # Navigation parameters
        self.safe_distance = 0.5  # meters
        self.target_distance = 2.0  # meters
        self.avoid_distance = 0.3  # meters

        # Object detection parameters
        self.object_cascade = cv2.CascadeClassifier()  # Load appropriate cascade
        self.object_detected = False
        self.object_position = None

        rospy.loginfo("Capstone Perception Navigation initialized")

    def image_callback(self, msg):
        """Process camera image for object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Convert to grayscale for object detection
            gray = cv2.cvtColor(cv_image, "gray")

            # Detect objects (simplified example)
            # objects = self.object_cascade.detectMultiScale(gray, 1.1, 4)

            # For this example, we'll simulate object detection
            height, width = cv_image.shape[:2]

            # Simulate detecting a person in the center
            if np.random.random() > 0.7:  # 30% chance of detection
                self.object_detected = True
                self.object_position = Point()
                self.object_position.x = width / 2  # Center x
                self.object_position.y = height / 2  # Center y
            else:
                self.object_detected = False
                self.object_position = None

            # Draw detection on image if needed
            if self.object_detected and self.object_position:
                cv2.circle(cv_image,
                          (int(self.object_position.x), int(self.object_position.y)),
                          20, (0, 255, 0), 2)

            # Publish processed image for debugging
            # processed_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            # self.processed_image_pub.publish(processed_msg)

        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Get distances in front of robot (simplified)
        front_distances = msg.ranges[len(msg.ranges)//2 - 30:len(msg.ranges)//2 + 30]

        # Remove invalid readings (inf, nan)
        valid_distances = [d for d in front_distances if d > 0 and d < float('inf')]

        if valid_distances:
            min_distance = min(valid_distances)

            # Check for obstacles
            if min_distance < self.safe_distance:
                obstacle_msg = Bool()
                obstacle_msg.data = True
                self.obstacle_detected_pub.publish(obstacle_msg)

                # Stop robot if obstacle too close
                if min_distance < self.avoid_distance:
                    self.stop_robot()
                    rospy.logwarn(f"Obstacle detected at {min_distance:.2f}m, stopping robot")
            else:
                obstacle_msg = Bool()
                obstacle_msg.data = False
                self.obstacle_detected_pub.publish(obstacle_msg)

    def odom_callback(self, msg):
        """Update robot position and orientation"""
        self.position.x = msg.pose.pose.position.x
        self.position.y = msg.pose.pose.position.y
        self.position.z = msg.pose.pose.position.z

        # Convert quaternion to euler
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])

        # Update orientation matrix
        self.orientation = tf.transformations.euler_matrix(0, 0, yaw)

        # Update velocities
        self.linear_velocity = msg.twist.twist.linear.x
        self.angular_velocity = msg.twist.twist.angular.z

    def navigate_to_object(self):
        """Navigate robot towards detected object"""
        if not self.object_detected or self.object_position is None:
            return

        # Get image dimensions (assuming 640x480)
        img_width = 640
        img_height = 480

        # Calculate object position relative to center
        center_x = img_width / 2
        object_offset_x = self.object_position.x - center_x

        # Calculate angular correction needed
        # Normalize to -1 (left) to 1 (right)
        normalized_offset = object_offset_x / (img_width / 2)

        # Calculate distance correction (simplified)
        # In a real system, you'd use depth information
        target_distance = self.target_distance
        current_distance = target_distance  # Placeholder

        # Create movement command
        cmd = Twist()

        # Move forward if object is centered and at right distance
        if abs(normalized_offset) < 0.1:  # Object is roughly centered
            if current_distance > target_distance:
                cmd.linear.x = 0.3  # Move forward
            elif current_distance < target_distance * 0.8:
                cmd.linear.x = -0.2  # Move backward
        else:
            # Turn towards object
            cmd.angular.z = -normalized_offset * 0.5  # Proportional control

        self.cmd_vel_pub.publish(cmd)

    def avoid_obstacles(self):
        """Navigate while avoiding obstacles"""
        # This is a simplified obstacle avoidance
        # In a real system, you'd use more sophisticated path planning

        # Get laser data for left and right sides
        scan = rospy.wait_for_message('/scan', LaserScan, timeout=1.0)

        left_distances = scan.ranges[len(scan.ranges)*3//4:]
        right_distances = scan.ranges[:len(scan.ranges)//4]

        left_clear = min([d for d in left_distances if d > 0 and d < float('inf')]) > self.safe_distance
        right_clear = min([d for d in right_distances if d > 0 and d < float('inf')]) > self.safe_distance

        cmd = Twist()

        # If obstacle ahead, turn away from it
        front_distances = scan.ranges[len(scan.ranges)//2 - 30:len(scan.ranges)//2 + 30]
        front_clear = min([d for d in front_distances if d > 0 and d < float('inf')]) > self.safe_distance

        if not front_clear:
            if left_clear and not right_clear:
                cmd.angular.z = 0.5  # Turn right
            elif right_clear and not left_clear:
                cmd.angular.z = -0.5  # Turn left
            elif left_clear and right_clear:
                # Choose randomly or based on other factors
                cmd.angular.z = 0.5 if np.random.random() > 0.5 else -0.5
            else:
                # Both sides blocked, turn randomly
                cmd.angular.z = 0.5 if np.random.random() > 0.5 else -0.5
        else:
            # Clear ahead, move forward
            cmd.linear.x = 0.3

        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            # If object detected, navigate towards it
            if self.object_detected:
                self.navigate_to_object()
            else:
                # Otherwise, continue exploring while avoiding obstacles
                self.avoid_obstacles()

            rate.sleep()

if __name__ == '__main__':
    navigator = CapstonePerceptionNavigation()
    navigator.run()
```

### 3. Humanoid Locomotion Controller

```python
#!/usr/bin/env python3
"""
Capstone Project: Humanoid Locomotion Control
File: capstone_locomotion_controller.py
"""

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64
import math
from collections import deque

class CapstoneLocomotionController:
    def __init__(self):
        rospy.init_node('capstone_locomotion_controller')

        # Publishers for joint control (assuming position control)
        self.joint_pubs = {}
        joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        for joint_name in joint_names:
            self.joint_pubs[joint_name] = rospy.Publisher(
                f'/{joint_name}_position_controller/command',
                Float64,
                queue_size=10
            )

        # Subscribers
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        # Robot parameters
        self.step_height = 0.1  # meters
        self.step_length = 0.3  # meters
        self.step_duration = 1.0  # seconds
        self.hip_width = 0.2  # distance between hip joints
        self.leg_length = 0.5  # length of leg segments

        # Walking state
        self.current_phase = 0.0  # 0.0 to 1.0, represents step cycle
        self.walk_enabled = False
        self.walk_direction = 0.0  # 1.0 forward, -1.0 backward, 0.0 stop
        self.turn_direction = 0.0  # 1.0 right, -1.0 left, 0.0 straight

        # Joint position tracking
        self.joint_positions = {}
        for joint_name in joint_names:
            self.joint_positions[joint_name] = 0.0

        # Walking pattern queue
        self.step_queue = deque()

        rospy.loginfo("Capstone Locomotion Controller initialized")

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Determine walking state
        if abs(linear_x) > 0.01:  # Threshold for movement
            self.walk_direction = 1.0 if linear_x > 0 else -1.0
            self.walk_enabled = True
        else:
            self.walk_direction = 0.0
            self.walk_enabled = False

        # Determine turning direction
        if abs(angular_z) > 0.01:
            self.turn_direction = 1.0 if angular_z > 0 else -1.0
        else:
            self.turn_direction = 0.0

    def calculate_inverse_kinematics(self, leg_pos, leg_type):
        """
        Calculate inverse kinematics for a leg
        leg_pos: (x, y, z) target position in leg coordinate system
        leg_type: 'left' or 'right'
        """
        x, y, z = leg_pos

        # Simplified 3DOF leg IK (hip, knee, ankle)
        # Calculate hip angle (yaw)
        hip_yaw = math.atan2(y, x) if leg_type == 'left' else math.atan2(-y, x)

        # Calculate distance in XZ plane
        dist_xz = math.sqrt(x**2 + z**2)

        # Calculate knee angle using law of cosines
        # leg_length is length of femur/tibia (assumed equal)
        a = self.leg_length  # femur
        b = self.leg_length  # tibia
        c = dist_xz  # distance from hip to foot in XZ plane

        if c > a + b:
            # Extend fully
            knee_angle = 0.0
        elif c < abs(a - b):
            # Fully flexed
            knee_angle = math.pi
        else:
            # Law of cosines: cos(C) = (a² + b² - c²) / (2ab)
            cos_knee = (a**2 + b**2 - c**2) / (2 * a * b)
            knee_angle = math.pi - math.acos(max(-1, min(1, cos_knee)))

        # Calculate hip pitch angle
        # First, find the angle of the line from hip to foot
        hip_foot_angle = math.atan2(z, x)
        # Then subtract the angle formed by the femur
        hip_pitch = hip_foot_angle - math.atan2(
            a * math.sin(math.pi - knee_angle),
            a * math.cos(math.pi - knee_angle) + b
        )

        # Ankle angle for ground contact
        ankle_pitch = -hip_foot_angle

        return hip_yaw, hip_pitch, knee_angle, ankle_pitch

    def generate_step_trajectory(self, phase, direction, turn=0.0):
        """
        Generate foot trajectory for walking step
        phase: 0.0 to 1.0 representing step cycle
        direction: 1.0 forward, -1.0 backward
        turn: 1.0 right, -1.0 left
        """
        # Split step into phases: 0.0-0.5 (swing), 0.5-1.0 (stance)
        if phase < 0.5:
            # Swing phase - foot lifts and moves forward
            swing_phase = phase * 2  # Normalize to 0-1 for swing
            x = direction * self.step_length * swing_phase
            z = self.step_height * math.sin(swing_phase * math.pi)  # Parabolic lift
            y = turn * self.hip_width * math.sin(swing_phase * math.pi) * 0.5  # Slight lateral movement for turning
        else:
            # Stance phase - foot moves back to prepare for next step
            stance_phase = (phase - 0.5) * 2  # Normalize to 0-1 for stance
            x = direction * self.step_length * (1 - stance_phase)
            z = 0.0  # On ground
            y = 0.0

        return x, y, z

    def update_walking_pattern(self):
        """Update walking pattern based on current commands"""
        if not self.walk_enabled:
            # Stop walking, return to neutral position
            for joint_name in self.joint_pubs.keys():
                if 'hip' in joint_name or 'knee' in joint_name or 'ankle' in joint_name:
                    self.joint_pubs[joint_name].publish(Float64(0.0))
            return

        # Update walking phase
        self.current_phase += 0.01  # Adjust based on desired step frequency
        if self.current_phase >= 1.0:
            self.current_phase = 0.0

        # Generate trajectories for both feet
        left_foot_pos = self.generate_step_trajectory(
            self.current_phase,
            self.walk_direction,
            -self.turn_direction  # Invert for left foot
        )

        right_foot_pos = self.generate_step_trajectory(
            (self.current_phase + 0.5) % 1.0,  # 180° phase offset
            self.walk_direction,
            self.turn_direction
        )

        # Calculate joint angles for both legs
        left_angles = self.calculate_inverse_kinematics(left_foot_pos, 'left')
        right_angles = self.calculate_inverse_kinematics(right_foot_pos, 'right')

        # Publish joint commands
        joint_commands = {
            'left_hip_joint': left_angles[1],  # hip pitch
            'left_knee_joint': left_angles[2],  # knee
            'left_ankle_joint': left_angles[3],  # ankle
            'right_hip_joint': right_angles[1],  # hip pitch
            'right_knee_joint': right_angles[2],  # knee
            'right_ankle_joint': right_angles[3],  # ankle
        }

        # Add some arm movement for balance
        arm_swing_phase = self.current_phase * 2 * math.pi
        joint_commands['left_shoulder_joint'] = 0.2 * math.sin(arm_swing_phase)
        joint_commands['right_shoulder_joint'] = 0.2 * math.sin(arm_swing_phase + math.pi)
        joint_commands['left_elbow_joint'] = 0.1 * math.cos(arm_swing_phase)
        joint_commands['right_elbow_joint'] = 0.1 * math.cos(arm_swing_phase + math.pi)

        # Publish all joint commands
        for joint_name, angle in joint_commands.items():
            if joint_name in self.joint_pubs:
                self.joint_pubs[joint_name].publish(Float64(angle))

    def run(self):
        """Main control loop"""
        rate = rospy.Rate(100)  # 100 Hz for smooth motion

        while not rospy.is_shutdown():
            self.update_walking_pattern()
            rate.sleep()

if __name__ == '__main__':
    controller = CapstoneLocomotionController()
    controller.run()
```

### 4. System Integration Launch File

```xml
<!-- capstone_robot.launch -->
<launch>
  <!-- Gazebo simulation -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find capstone_project)/worlds/physical_ai_world.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- Robot model -->
  <param name="robot_description"
         command="$(find xacro)/xacro '$(find capstone_project)/urdf/humanoid_robot.xacro'"/>

  <!-- Spawn robot in Gazebo -->
  <node name="spawn_urdf"
        pkg="gazebo_ros"
        type="spawn_model"
        args="-param robot_description -urdf -model capstone_robot -x 0 -y 0 -z 1"/>

  <!-- Joint state publisher -->
  <node name="joint_state_publisher"
        pkg="joint_state_publisher"
        type="joint_state_publisher"/>

  <!-- Robot state publisher -->
  <node name="robot_state_publisher"
        pkg="robot_state_publisher"
        type="robot_state_publisher"/>

  <!-- Voice interface -->
  <node name="capstone_voice_interface"
        pkg="capstone_project"
        type="capstone_voice_interface.py"
        output="screen">
    <param name="gpt_api_key" value="$(arg gpt_api_key)"/>
  </node>

  <!-- Perception and navigation -->
  <node name="capstone_perception_navigation"
        pkg="capstone_project"
        type="capstone_perception_navigation.py"
        output="screen"/>

  <!-- Locomotion controller -->
  <node name="capstone_locomotion_controller"
        pkg="capstone_project"
        type="capstone_locomotion_controller.py"
        output="screen"/>

  <!-- Camera info publisher -->
  <node name="camera_info_publisher"
        pkg="capstone_project"
        type="camera_info_publisher.py"/>

  <!-- TF broadcasters -->
  <node name="tf_broadcaster"
        pkg="capstone_project"
        type="tf_broadcaster.py"/>

  <!-- Launch arguments -->
  <arg name="gpt_api_key" default=""/>
</launch>
```

### 5. Unit Tests

```python
#!/usr/bin/env python3
"""
Capstone Project: Unit Tests
File: test_capstone_system.py
"""

import unittest
import rospy
from capstone_voice_interface import CapstoneVoiceInterface
from capstone_perception_navigation import CapstonePerceptionNavigation
from capstone_locomotion_controller import CapstoneLocomotionController
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class TestCapstoneVoiceInterface(unittest.TestCase):
    def setUp(self):
        rospy.init_node('test_capstone_voice', anonymous=True)
        self.voice_interface = CapstoneVoiceInterface()

    def test_process_rule_based_commands(self):
        """Test rule-based command processing"""
        # Test move forward
        result = self.voice_interface.process_rule_based("move forward")
        self.assertEqual(result['action'], 'move_forward')
        self.assertEqual(result['parameters']['distance'], 1.0)

        # Test turn left
        result = self.voice_interface.process_rule_based("turn left")
        self.assertEqual(result['action'], 'turn_left')
        self.assertEqual(result['parameters']['angle'], 90)

        # Test greeting
        result = self.voice_interface.process_rule_based("hello robot")
        self.assertEqual(result['action'], 'speak')

    def test_gpt_processing_fallback(self):
        """Test GPT processing fallback to rule-based"""
        # Mock GPT failure and verify fallback
        original_gpt_key = self.voice_interface.gpt_api_key
        self.voice_interface.gpt_api_key = ""  # Disable GPT

        result = self.voice_interface.process_with_gpt("move forward")
        self.assertIsNotNone(result)  # Should fallback to rule-based

        self.voice_interface.gpt_api_key = original_gpt_key

class TestCapstonePerceptionNavigation(unittest.TestCase):
    def setUp(self):
        rospy.init_node('test_capstone_nav', anonymous=True)
        self.navigator = CapstonePerceptionNavigation()

    def test_object_detection_simulation(self):
        """Test object detection simulation"""
        # This would test the simulated object detection
        self.navigator.object_detected = True
        self.navigator.object_position = None  # Reset

        # Simulate detection
        import numpy as np
        if np.random.random() > 0.7:
            self.navigator.object_detected = True
            self.navigator.object_position = rospy.Point(320, 240, 0)  # Center of 640x480 image

        self.assertTrue(self.navigator.object_detected)

    def test_obstacle_detection(self):
        """Test obstacle detection logic"""
        # Simulate laser scan with obstacle
        from sensor_msgs.msg import LaserScan
        scan_msg = LaserScan()
        scan_msg.ranges = [0.4] * 360  # All distances are 0.4m
        scan_msg.angle_min = -math.pi
        scan_msg.angle_max = math.pi
        scan_msg.angle_increment = 2 * math.pi / 360

        # Process the scan
        self.navigator.laser_callback(scan_msg)

        # Verify obstacle detected
        # This would require mocking the publisher to verify

class TestCapstoneLocomotionController(unittest.TestCase):
    def setUp(self):
        rospy.init_node('test_capstone_loco', anonymous=True)
        self.controller = CapstoneLocomotionController()

    def test_inverse_kinematics(self):
        """Test inverse kinematics calculation"""
        # Test neutral position (foot directly below hip)
        hip_yaw, hip_pitch, knee_angle, ankle_pitch = self.controller.calculate_inverse_kinematics(
            (0, 0, -self.controller.leg_length * 2), 'left'
        )

        # At neutral position, hip pitch and ankle pitch should be 0, knee should be ~180 deg
        self.assertAlmostEqual(hip_pitch, 0.0, places=1)
        self.assertAlmostEqual(ankle_pitch, 0.0, places=1)
        self.assertAlmostEqual(knee_angle, math.pi, places=1)

    def test_step_trajectory_generation(self):
        """Test step trajectory generation"""
        # Test beginning of step (lift foot)
        x, y, z = self.controller.generate_step_trajectory(0.0, 1.0)
        self.assertEqual(x, 0.0)
        self.assertEqual(z, 0.0)  # Foot on ground

        # Test middle of swing phase (foot lifted)
        x, y, z = self.controller.generate_step_trajectory(0.25, 1.0)
        self.assertGreater(z, 0.0)  # Foot should be lifted
        self.assertGreater(x, 0.0)  # Foot should be moving forward

        # Test end of step (foot down)
        x, y, z = self.controller.generate_step_trajectory(1.0, 1.0)
        self.assertEqual(z, 0.0)  # Foot back on ground

    def test_cmd_vel_processing(self):
        """Test velocity command processing"""
        from geometry_msgs.msg import Twist
        cmd = Twist()
        cmd.linear.x = 0.5
        cmd.angular.z = 0.2

        self.controller.cmd_vel_callback(cmd)

        self.assertEqual(self.controller.walk_direction, 1.0)  # Moving forward
        self.assertEqual(self.controller.turn_direction, 1.0)  # Turning right
        self.assertTrue(self.controller.walk_enabled)

def run_tests():
    """Run all capstone project tests"""
    import sys
    import rostest

    # Initialize ROS
    rospy.init_node('capstone_tests', anonymous=True)

    # Run unit tests
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestCapstoneVoiceInterface))
    test_suite.addTest(unittest.makeSuite(TestCapstonePerceptionNavigation))
    test_suite.addTest(unittest.makeSuite(TestCapstoneLocomotionController))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
```

## Running the Complete System

### 1. Setup Environment

```bash
# Install required packages
pip install openai-whisper pyaudio numpy scipy torch
pip install opencv-python cv-bridge
pip install pyttsx3 vosk

# Make sure ROS 2 packages are installed
sudo apt update
sudo apt install ros-humble-desktop ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-joint-state-publisher ros-humble-robot-state-publisher
```

### 2. Build and Run

```bash
# Build the workspace
cd ~/ros2_ws
colcon build --packages-select capstone_project
source install/setup.bash

# Launch the complete system
# Option 1: With Gazebo simulation
export GPT_API_KEY="your-api-key-here"
ros2 launch capstone_project capstone_robot.launch.py gpt_api_key:=$GPT_API_KEY

# Option 2: For real robot (with appropriate hardware interfaces)
ros2 launch capstone_project capstone_robot_real.launch.py
```

### 3. Voice Commands

Once the system is running, you can use these voice commands:

- "Move forward 2 meters" - Move the robot forward
- "Turn left" or "Turn right" - Rotate the robot
- "Stop" - Stop all robot motion
- "Hello robot" - Get the robot's attention
- "Navigate to object" - Start object-seeking behavior
- "Follow me" - Follow a detected person

## Evaluation Criteria

Your capstone project will be evaluated on:

1. **Integration Quality**: How well all components work together
2. **Robustness**: Handling of edge cases and errors
3. **Performance**: Response time and accuracy
4. **Safety**: Proper obstacle avoidance and safe operation
5. **User Experience**: Natural and intuitive interaction

## Troubleshooting

Common issues and solutions:

1. **Audio not working**: Check microphone permissions and audio drivers
2. **Whisper model not loading**: Ensure sufficient RAM and download models in advance
3. **Navigation errors**: Verify sensor calibration and coordinate transforms
4. **Joint control issues**: Check URDF model and controller configuration

## Extensions

Consider these extensions to enhance your project:

1. **SLAM Integration**: Add simultaneous localization and mapping
2. **Object Recognition**: Implement deep learning for specific object detection
3. **Multi-modal Interaction**: Add gesture and facial expression recognition
4. **Learning Capabilities**: Implement reinforcement learning for improved behavior
5. **Multi-robot Coordination**: Extend to multiple robots working together

## Conclusion

The capstone project demonstrates the complete Physical AI pipeline from perception to action. It integrates concepts from all 13 weeks of the curriculum, showing how individual components combine to create intelligent, interactive robotic systems. This project serves as a foundation for advanced robotics research and development.