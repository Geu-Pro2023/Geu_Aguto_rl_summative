#!/usr/bin/env python3
"""
Simplified training script for demonstration without heavy dependencies
"""

import sys
import numpy as np
import random
import json
from collections import defaultdict, deque
import matplotlib.pyplot as plt

sys.path.append('environment')
from environment.custom_env import CattleMonitoringEnv

class SimpleQAgent:
    """Simple Q-Learning agent for demonstration"""
    
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.95, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Simple Q-table (discretized states)
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
    def discretize_state(self, state):
        """Convert continuous state to discrete for Q-table"""
        # Simple discretization - just use agent position and basic info
        agent_pos = tuple(state[:2].astype(int))
        cattle_nearby = int(np.sum(np.linalg.norm(state[2:8].reshape(-1, 2) - state[:2], axis=1) < 2))
        return (agent_pos[0], agent_pos[1], cattle_nearby)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        discrete_state = self.discretize_state(state)
        
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[discrete_next_state])
            
        self.q_table[discrete_state][action] += self.lr * (target - self.q_table[discrete_state][action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SimpleREINFORCE:
    """Simple REINFORCE implementation"""
    
    def __init__(self, state_size, action_size, lr=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        # Simple policy network weights (linear)
        self.weights = np.random.randn(state_size, action_size) * 0.1
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
    def softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def act(self, state):
        """Choose action using policy"""
        logits = np.dot(state, self.weights)
        probs = self.softmax(logits)
        action = np.random.choice(self.action_size, p=probs)
        
        self.episode_states.append(state)
        self.episode_actions.append(action)
        
        return action
    
    def learn(self):
        """Update policy using REINFORCE"""
        if len(self.episode_rewards) == 0:
            return
            
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + 0.99 * G
            returns.insert(0, G)
            
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Update weights
        for i, (state, action, G) in enumerate(zip(self.episode_states, self.episode_actions, returns)):
            logits = np.dot(state, self.weights)
            probs = self.softmax(logits)
            
            # Gradient update
            grad = np.zeros_like(self.weights)
            grad[:, action] = state * (1 - probs[action]) * G
            for a in range(self.action_size):
                if a != action:
                    grad[:, a] = -state * probs[a] * G
                    
            self.weights += self.lr * grad
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def store_reward(self, reward):
        """Store reward for episode"""
        self.episode_rewards.append(reward)

def train_agent(agent_type="q_learning", episodes=1000):
    """Train an agent"""
    env = CattleMonitoringEnv()
    
    if agent_type == "q_learning":
        agent = SimpleQAgent(env.observation_space.shape[0], env.action_space.n)
    else:  # reinforce
        agent = SimpleREINFORCE(env.observation_space.shape[0], env.action_space.n)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Training {agent_type.upper()} for {episodes} episodes...")
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 200:  # Max steps per episode
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            if agent_type == "q_learning":
                agent.learn(state, action, reward, next_state, terminated or truncated)
            else:
                agent.store_reward(reward)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        if agent_type == "reinforce":
            agent.learn()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Check success
        if info['cattle_registered'] >= 2 and info['alerts_sent'] >= 1:
            success_count += 1
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = success_count / (episode + 1) * 100
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Success Rate = {success_rate:.1f}%")
    
    return {
        'algorithm': agent_type.upper(),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_success_rate': success_count / episodes * 100,
        'mean_reward': np.mean(episode_rewards[-100:]),
        'agent': agent
    }

def evaluate_agent(agent, agent_type, episodes=10):
    """Evaluate trained agent"""
    env = CattleMonitoringEnv()
    
    episode_rewards = []
    success_count = 0
    
    print(f"\\nEvaluating {agent_type.upper()}...")
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        # Disable exploration for evaluation
        if hasattr(agent, 'epsilon'):
            old_epsilon = agent.epsilon
            agent.epsilon = 0
        
        while steps < 200:
            action = agent.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        
        if info['cattle_registered'] >= 2 and info['alerts_sent'] >= 1:
            success_count += 1
            
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, "
              f"Registered = {info['cattle_registered']}, Alerts = {info['alerts_sent']}")
        
        # Restore exploration
        if hasattr(agent, 'epsilon'):
            agent.epsilon = old_epsilon
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': success_count / episodes * 100,
        'episode_rewards': episode_rewards
    }
    
    print(f"Results: Mean Reward = {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Success Rate = {results['success_rate']:.1f}%")
    
    return results

def create_comparison_plot(results_dict):
    """Create comparison plots"""
    algorithms = list(results_dict.keys())
    mean_rewards = [results_dict[algo]['mean_reward'] for algo in algorithms]
    success_rates = [results_dict[algo]['final_success_rate'] for algo in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean rewards
    bars1 = ax1.bar(algorithms, mean_rewards, color=['blue', 'green'])
    ax1.set_title('Mean Reward Comparison')
    ax1.set_ylabel('Mean Reward')
    ax1.set_xlabel('Algorithm')
    
    for bar, reward in zip(bars1, mean_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{reward:.1f}', ha='center', va='bottom')
    
    # Success rates
    bars2 = ax2.bar(algorithms, success_rates, color=['blue', 'green'])
    ax2.set_title('Success Rate Comparison')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylim(0, max(success_rates) * 1.2)
    
    for bar, rate in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('rl_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Comparison plot saved as 'rl_comparison_results.png'")

def main():
    """Main training and evaluation pipeline"""
    print("="*60)
    print("CATTLE MONITORING RL DEMONSTRATION")
    print("="*60)
    
    # Train both algorithms
    results = {}
    
    # Train Q-Learning (Value-Based)
    print("\\n1. Training Q-Learning (Value-Based Method)")
    print("-" * 40)
    q_results = train_agent("q_learning", episodes=500)
    results['Q-Learning'] = q_results
    
    # Evaluate Q-Learning
    q_eval = evaluate_agent(q_results['agent'], "q_learning")
    results['Q-Learning'].update(q_eval)
    
    # Train REINFORCE (Policy Gradient)
    print("\\n2. Training REINFORCE (Policy Gradient Method)")
    print("-" * 40)
    reinforce_results = train_agent("reinforce", episodes=500)
    results['REINFORCE'] = reinforce_results
    
    # Evaluate REINFORCE
    reinforce_eval = evaluate_agent(reinforce_results['agent'], "reinforce")
    results['REINFORCE'].update(reinforce_eval)
    
    # Compare results
    print("\\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    print(f"{'Algorithm':<12} {'Mean Reward':<12} {'Success Rate':<12}")
    print("-" * 40)
    for algo, result in results.items():
        print(f"{algo:<12} {result['mean_reward']:<12.2f} {result['success_rate']:<12.1f}%")
    
    # Create visualization
    create_comparison_plot(results)
    
    # Save results
    # Remove agent objects for JSON serialization
    save_results = {}
    for algo, result in results.items():
        save_results[algo] = {k: v for k, v in result.items() if k != 'agent'}
    
    with open('training_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print("✓ Results saved to 'training_results.json'")
    
    return results

if __name__ == "__main__":
    main()