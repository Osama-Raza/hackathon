---
title: "Reinforcement Learning Basics"
sidebar_label: "Reinforcement Learning Basics"
description: "Introduction to reinforcement learning for robotics applications"
---

# Reinforcement Learning Basics

## Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and aims to maximize cumulative rewards over time. In robotics, RL is particularly powerful for learning complex behaviors like manipulation, navigation, and control policies.

## Core Concepts

### 1. Agent-Environment Interaction

The fundamental RL framework consists of:
- **Agent**: The learning system that makes decisions
- **Environment**: The world the agent interacts with
- **State (s)**: The current situation of the environment
- **Action (a)**: What the agent can do
- **Reward (r)**: Feedback from the environment
- **Policy (π)**: Strategy that maps states to actions

### 2. Markov Decision Process (MDP)

RL problems are often modeled as MDPs with the tuple (S, A, P, R, γ):
- **S**: Set of possible states
- **A**: Set of possible actions
- **P**: State transition probabilities
- **R**: Reward function
- **γ**: Discount factor (0 ≤ γ ≤ 1)

## Types of RL Algorithms

### 1. Value-Based Methods

Value-based methods learn the value of states or state-action pairs:

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.95, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
```

### 2. Policy-Based Methods

Policy-based methods directly learn the policy function:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def get_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def update(self, log_probs, rewards, gamma=0.99):
        """Update policy using REINFORCE algorithm"""
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
```

### 3. Actor-Critic Methods

Actor-critic methods combine value and policy learning:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy network)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic (value network)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_extractor(state)

        # Actor: compute action probabilities
        action_logits = self.actor(features)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Critic: compute state value
        state_value = self.critic(features)

        return action_probs, state_value

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_action(self, state):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.model(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        return action.item(), action_dist.log_prob(action), state_value

    def update(self, log_probs, values, rewards, gamma=0.99):
        """Update actor-critic using advantage"""
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        values = torch.cat(values).squeeze()

        # Calculate advantages
        advantages = returns - values

        # Calculate losses
        actor_loss = -(torch.cat(log_probs) * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        # Update model
        self.optimizer.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        self.optimizer.step()
```

## Deep RL Algorithms for Robotics

### 1. Deep Q-Network (DQN)

DQN extends Q-learning with deep neural networks:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Main and target networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()

    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
```

### 2. Proximal Policy Optimization (PPO)

PPO is a popular policy gradient method for continuous control:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOActor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return mean, std

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(PPOCritic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, epochs=10):
        self.actor = PPOActor(state_dim, action_dim)
        self.critic = PPOCritic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.action_dim = action_dim

    def get_action(self, state):
        """Sample action from current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.actor(state_tensor)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action[0].numpy(), log_prob[0].numpy()

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy using PPO objective"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        advantages = torch.FloatTensor(advantages)

        for _ in range(self.epochs):
            # Actor update
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions)

            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                               1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic update
            values = self.critic(states)
            critic_loss = nn.MSELoss()(values, returns)

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
```

## RL for Robotics Applications

### 1. Continuous Control with Deep Deterministic Policy Gradient (DDPG)

DDPG is effective for continuous action spaces in robotics:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3,
                 tau=0.005, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.tau = tau
        self.gamma = gamma
        self.action_dim = action_dim

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(self, state, noise_scale=0.1):
        """Get action with noise for exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).squeeze(0).detach().numpy()

        # Add noise for exploration
        noise = np.random.normal(0, noise_scale, size=self.action_dim)
        action = np.clip(action + noise, -1, 1)

        return action

    def soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(),
                                     self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                   (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(),
                                     self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                   (1 - self.tau) * target_param.data)
```

### 2. Robotic Manipulation Example

```python
class RoboticManipulationEnvironment:
    def __init__(self):
        # Initialize robot simulation (e.g., PyBullet, Isaac Sim)
        self.state_dim = 14  # Example: 7 joint positions + 7 joint velocities
        self.action_dim = 7  # 7 joint torques
        self.max_episode_steps = 1000

    def reset(self):
        """Reset environment to initial state"""
        # Reset robot to initial configuration
        # Reset object positions
        # Return initial state
        pass

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Apply action to robot
        # Simulate physics
        # Calculate reward based on task (e.g., reaching, grasping)
        # Check termination conditions
        # Return (next_state, reward, done, info)
        pass

    def get_reward(self, state, action, next_state):
        """Calculate reward for manipulation task"""
        # Example: reacher task reward
        target_pos = self.target_position
        end_effector_pos = self.get_end_effector_position(next_state)

        distance = np.linalg.norm(target_pos - end_effector_pos)
        reward = -distance  # Negative distance as reward

        # Add bonus for getting close to target
        if distance < 0.1:
            reward += 10

        return reward
```

## Simulation-to-Real Transfer (Sim-to-Real)

### 1. Domain Randomization

```python
class DomainRandomization:
    def __init__(self):
        self.randomization_params = {
            'mass_range': [0.8, 1.2],  # 80% to 120% of original mass
            'friction_range': [0.1, 0.9],
            'lighting_range': [0.5, 2.0],
            'texture_randomization': True
        }

    def randomize_environment(self, env):
        """Randomize environment parameters"""
        # Randomize physical properties
        for obj in env.objects:
            obj.mass = np.random.uniform(*self.randomization_params['mass_range'])
            obj.friction = np.random.uniform(*self.randomization_params['friction_range'])

        # Randomize visual properties
        if self.randomization_params['texture_randomization']:
            env.set_random_textures()

        # Randomize lighting
        light_intensity = np.random.uniform(*self.randomization_params['lighting_range'])
        env.set_light_intensity(light_intensity)

    def train_with_randomization(self, agent, env, episodes=100000):
        """Train agent with domain randomization"""
        for episode in range(episodes):
            if episode % 100 == 0:  # Randomize every 100 episodes
                self.randomize_environment(env)

            # Train normally
            state = env.reset()
            done = False

            while not done:
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if hasattr(agent, 'replay'):
                    agent.replay()
```

### 2. System Identification

```python
class SystemID:
    def __init__(self):
        self.real_params = {}
        self.sim_params = {}

    def identify_real_system(self, robot):
        """Identify real robot parameters"""
        # Collect data from real robot
        # Estimate parameters (mass, friction, etc.)
        # Return identified parameters
        pass

    def adapt_sim_to_real(self, sim_env, real_params):
        """Adapt simulation to match real system"""
        # Update simulation parameters to match real robot
        for param, value in real_params.items():
            sim_env.set_parameter(param, value)
```

## Best Practices for Robotics RL

### 1. Reward Engineering

```python
class ShapedReward:
    def __init__(self):
        self.weights = {
            'distance': 1.0,
            'velocity': 0.1,
            'energy': 0.01,
            'smoothness': 0.05
        }

    def calculate_reward(self, state, action, next_state, target):
        """Calculate shaped reward for learning"""
        # Distance to goal
        dist_reward = -np.linalg.norm(target - self.get_position(next_state))

        # Velocity penalty to encourage smooth motion
        vel_norm = np.linalg.norm(self.get_velocity(next_state))
        vel_penalty = -self.weights['velocity'] * vel_norm

        # Energy efficiency
        energy_cost = -self.weights['energy'] * np.sum(np.abs(action))

        # Smoothness penalty
        smooth_penalty = -self.weights['smoothness'] * np.sum(np.abs(action))

        total_reward = (dist_reward + vel_penalty +
                       energy_cost + smooth_penalty)

        return total_reward
```

### 2. Exploration Strategies

```python
class ExplorationStrategy:
    def __init__(self, action_dim, noise_type='ou_process'):
        self.action_dim = action_dim
        self.noise_type = noise_type
        self.ou_process = np.zeros(action_dim)  # Ornstein-Uhlenbeck process

    def add_exploration_noise(self, action, episode, total_episodes):
        """Add exploration noise to action"""
        if self.noise_type == 'gaussian':
            noise_scale = max(0.1, 1.0 - episode / (total_episodes * 0.8))
            noise = np.random.normal(0, noise_scale, self.action_dim)
        elif self.noise_type == 'ou_process':
            # Ornstein-Uhlenbeck process for temporally correlated noise
            theta = 0.15
            sigma = 0.2
            noise = theta * (-self.ou_process) + sigma * np.random.normal(0, 1, self.action_dim)
            self.ou_process += noise
            noise = self.ou_process.copy()

        return np.clip(action + noise, -1, 1)
```

## Evaluation and Testing

### 1. Performance Metrics

```python
def evaluate_agent(agent, env, episodes=100):
    """Evaluate agent performance"""
    episode_returns = []
    episode_lengths = []
    success_rates = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        while not done:
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Check success condition
            if info.get('success', False):
                success_rates.append(1)
            elif done and not info.get('success', False):
                success_rates.append(0)

        episode_returns.append(total_reward)
        episode_lengths.append(step_count)

    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(success_rates) if success_rates else 0
    }
```

### 2. Safety Considerations

```python
class SafeRLWrapper:
    def __init__(self, env, safety_constraints):
        self.env = env
        self.safety_constraints = safety_constraints

    def step(self, action):
        """Step with safety checks"""
        # Check if action violates safety constraints
        if self._is_safe_action(action):
            next_state, reward, done, info = self.env.step(action)
        else:
            # Return safe fallback
            next_state, reward, done, info = self._safe_fallback()

        return next_state, reward, done, info

    def _is_safe_action(self, action):
        """Check if action is safe"""
        # Implement safety checks
        # e.g., joint limits, velocity limits, collision avoidance
        pass
```

Reinforcement learning provides powerful tools for learning complex robotic behaviors. By combining deep learning with RL algorithms, robots can learn to perform tasks that would be difficult to program explicitly. Success in RL for robotics requires careful consideration of reward design, exploration strategies, and simulation-to-real transfer techniques.