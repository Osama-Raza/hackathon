/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

module.exports = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
    },
    {
      type: 'category',
      label: 'Week 1-2: Introduction To Physical AI',
      items: [
        'week-1-2/foundations-physical-ai',
        'week-1-2/humanoid-robotics-landscape',
        'week-1-2/sensor-systems-overview',
        'week-1-2/system-architecture-diagram'
      ],
    },
    {
      type: 'category',
      label: 'Week 3-5: ROS 2 Fundamentals',
      items: [
        'week-3-5/ros2-architecture',
        'week-3-5/ros2-packages',
        {
          type: 'category',
          label: 'Examples',
          key: 'week3-5-examples',
          items: [
            'examples/ros2/talker-listener-example',
            'examples/ros2/launch-files-example',
            'examples/ros2/testing-ros2-examples'
          ]
        }
      ],
    },
    {
      type: 'category',
      label: 'Week 6-7: Robot Simulation with Gazebo',
      items: [
        'week-6-7/gazebo-setup',
        'week-6-7/urdf-sdf-robot',
        {
          type: 'category',
          label: 'Examples',
          key: 'week6-7-examples',
          items: [
            'examples/gazebo/simple-robot-urdf',
            'examples/gazebo/simple-world-file',
            'examples/gazebo/testing-gazebo-examples'
          ]
        }
      ],
    },
    {
      type: 'category',
      label: 'Week 8-10: NVIDIA Isaac Platform',
      items: [
        'week-8-10/isaac-sim-introduction',
        'week-8-10/ai-perception-pipelines',
        'week-8-10/reinforcement-learning-basics',
        {
          type: 'category',
          label: 'Examples',
          key: 'week8-10-examples',
          items: [
            'examples/isaac/isaac-sim-setup'
          ]
        }
      ],
    },
    {
      type: 'category',
      label: 'Week 11-12: Humanoid Robot Development',
      items: [
        'week-11-12/humanoid-kinematics',
        'week-11-12/bipedal-locomotion',
        'week-11-12/human-robot-interaction',
        'week-11-12/humanoid-kinematic-chain-diagram',
        {
          type: 'category',
          label: 'Examples',
          key: 'week11-12-examples',
          items: [
            'examples/humanoid/locomotion-controller'
          ]
        }
      ],
    },
    {
      type: 'category',
      label: 'Week 13: Conversational Robotics',
      items: [
        'week-13/gpt-integration-for-robots',
        'week-13/voice-to-action-pipeline',
        'week-13/capstone-project-guide',
        'week-13/capstone-architecture-diagram',
        {
          type: 'category',
          label: 'Examples',
          key: 'week13-examples',
          items: [
            'examples/voice/voice-command-processor'
          ]
        }
      ],
    },
    {
      type: 'category',
      label: 'Reference Materials',
      items: [
        'reference/isaac-sim-requirements',
        'reference/hardware-requirements',
        'reference/installation-guide',
        'reference/troubleshooting'
      ],
    }
  ],
};