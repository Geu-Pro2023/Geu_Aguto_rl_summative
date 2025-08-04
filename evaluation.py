"""
Comprehensive evaluation and comparison of all RL algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C
import torch
import torch.nn.functional as F
from environment.custom_env import CattleVerificationEnv
from training.reinforce_training import PolicyNetwork, flatten_observation
import os
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_sb3_model(self, model_path, model_type, num_episodes=10, max_steps=500):
        """Evaluate Stable Baselines3 models (DQN, PPO, A2C)"""
        env = CattleVerificationEnv(render_mode=None, size=10)
        
        try:
            model = globals()[model_type.upper()].load(model_path)
            
            episode_rewards = []
            episode_lengths = []
            cattle_verified = []
            success_count = 0
            
            for episode in range(num_episodes):
                obs, _ = env.reset()
                total_reward = 0
                step = 0
                
                for step in range(max_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(total_reward)
                episode_lengths.append(step + 1)
                cattle_verified.append(info['verified_count'])
                
                if total_reward > 0:
                    success_count += 1
            
            results = {
                'algorithm': model_type.upper(),
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'avg_length': np.mean(episode_lengths),
                'avg_verified': np.mean(cattle_verified),
                'success_rate': success_count / num_episodes * 100,
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'cattle_verified': cattle_verified
            }
            
            self.results[model_type.upper()] = results
            return results
            
        finally:
            env.close()
    
    def evaluate_reinforce_model(self, model_path, num_episodes=10, max_steps=500):
        """Evaluate REINFORCE model"""
        env = CattleVerificationEnv(render_mode=None, size=10)
        
        try:
            # Load REINFORCE model
            state_dim = 2 + 2 + 8 + 4
            action_dim = 4
            policy_net = PolicyNetwork(state_dim, action_dim)
            policy_net.load_state_dict(torch.load(model_path))
            policy_net.eval()
            
            episode_rewards = []
            episode_lengths = []
            cattle_verified = []
            success_count = 0
            
            for episode in range(num_episodes):
                obs, _ = env.reset()
                state = flatten_observation(obs)
                total_reward = 0
                step = 0
                
                for step in range(max_steps):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        probs = policy_net(state_tensor)
                        action = torch.argmax(probs).item()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                    
                    state = flatten_observation(obs)
                
                episode_rewards.append(total_reward)
                episode_lengths.append(step + 1)
                cattle_verified.append(info['verified_count'])
                
                if total_reward > 0:
                    success_count += 1
            
            results = {
                'algorithm': 'REINFORCE',
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'avg_length': np.mean(episode_lengths),
                'avg_verified': np.mean(cattle_verified),
                'success_rate': success_count / num_episodes * 100,
                'max_reward': np.max(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'cattle_verified': cattle_verified
            }
            
            self.results['REINFORCE'] = results
            return results
            
        finally:
            env.close()
    
    def compare_algorithms(self):
        """Generate comprehensive comparison of all algorithms"""
        if not self.results:
            print("No evaluation results available. Run evaluations first.")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for algo, results in self.results.items():
            comparison_data.append({
                'Algorithm': algo,
                'Avg Reward': f"{results['avg_reward']:.2f} ± {results['std_reward']:.2f}",
                'Success Rate (%)': f"{results['success_rate']:.1f}",
                'Avg Steps': f"{results['avg_length']:.1f}",
                'Avg Cattle Verified': f"{results['avg_verified']:.1f}/4",
                'Best Reward': f"{results['max_reward']:.2f}",
                'Worst Reward': f"{results['min_reward']:.2f}"
            })\n        \n        df = pd.DataFrame(comparison_data)\n        print(\"\\n\" + \"=\"*80)\n        print(\"ALGORITHM PERFORMANCE COMPARISON\")\n        print(\"=\"*80)\n        print(df.to_string(index=False))\n        print(\"=\"*80)\n        \n        return df\n    \n    def plot_comparison(self, save_path=\"evaluation_results.png\"):\n        \"\"\"Create visualization comparing all algorithms\"\"\"\n        if not self.results:\n            print(\"No evaluation results available. Run evaluations first.\")\n            return\n        \n        fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n        fig.suptitle('RL Algorithm Comparison - Cattle Verification System', fontsize=16)\n        \n        algorithms = list(self.results.keys())\n        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']\n        \n        # Average Reward Comparison\n        avg_rewards = [self.results[algo]['avg_reward'] for algo in algorithms]\n        std_rewards = [self.results[algo]['std_reward'] for algo in algorithms]\n        \n        axes[0, 0].bar(algorithms, avg_rewards, yerr=std_rewards, \n                       color=colors[:len(algorithms)], alpha=0.7, capsize=5)\n        axes[0, 0].set_title('Average Episode Reward')\n        axes[0, 0].set_ylabel('Reward')\n        axes[0, 0].grid(True, alpha=0.3)\n        \n        # Success Rate Comparison\n        success_rates = [self.results[algo]['success_rate'] for algo in algorithms]\n        axes[0, 1].bar(algorithms, success_rates, color=colors[:len(algorithms)], alpha=0.7)\n        axes[0, 1].set_title('Success Rate')\n        axes[0, 1].set_ylabel('Success Rate (%)')\n        axes[0, 1].grid(True, alpha=0.3)\n        \n        # Average Episode Length\n        avg_lengths = [self.results[algo]['avg_length'] for algo in algorithms]\n        axes[1, 0].bar(algorithms, avg_lengths, color=colors[:len(algorithms)], alpha=0.7)\n        axes[1, 0].set_title('Average Episode Length')\n        axes[1, 0].set_ylabel('Steps')\n        axes[1, 0].grid(True, alpha=0.3)\n        \n        # Cattle Verification Rate\n        avg_verified = [self.results[algo]['avg_verified'] for algo in algorithms]\n        axes[1, 1].bar(algorithms, avg_verified, color=colors[:len(algorithms)], alpha=0.7)\n        axes[1, 1].set_title('Average Cattle Verified')\n        axes[1, 1].set_ylabel('Cattle Count')\n        axes[1, 1].set_ylim(0, 4)\n        axes[1, 1].grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.savefig(save_path, dpi=300, bbox_inches='tight')\n        plt.show()\n        print(f\"Comparison plot saved to {save_path}\")\n    \n    def save_results(self, filename=\"evaluation_results.json\"):\n        \"\"\"Save evaluation results to JSON file\"\"\"\n        # Convert numpy arrays to lists for JSON serialization\n        json_results = {}\n        for algo, results in self.results.items():\n            json_results[algo] = {\n                'algorithm': results['algorithm'],\n                'avg_reward': float(results['avg_reward']),\n                'std_reward': float(results['std_reward']),\n                'avg_length': float(results['avg_length']),\n                'avg_verified': float(results['avg_verified']),\n                'success_rate': float(results['success_rate']),\n                'max_reward': float(results['max_reward']),\n                'min_reward': float(results['min_reward']),\n                'episode_rewards': [float(x) for x in results['episode_rewards']],\n                'episode_lengths': [int(x) for x in results['episode_lengths']],\n                'cattle_verified': [int(x) for x in results['cattle_verified']]\n            }\n        \n        json_results['evaluation_date'] = datetime.now().isoformat()\n        \n        with open(filename, 'w') as f:\n            json.dump(json_results, f, indent=2)\n        \n        print(f\"Results saved to {filename}\")\n\ndef main():\n    \"\"\"Run comprehensive evaluation of all algorithms\"\"\"\n    print(\"Starting comprehensive evaluation of all RL algorithms...\")\n    \n    evaluator = ModelEvaluator()\n    \n    # Evaluate all algorithms\n    algorithms_to_evaluate = [\n        (\"models/dqn/dqn_cattle_final\", \"dqn\"),\n        (\"models/ppo/ppo_cattle_final\", \"ppo\"),\n        (\"models/a2c/a2c_cattle_final\", \"a2c\")\n    ]\n    \n    # Evaluate SB3 models\n    for model_path, model_type in algorithms_to_evaluate:\n        if os.path.exists(model_path + \".zip\"):\n            print(f\"\\nEvaluating {model_type.upper()}...\")\n            evaluator.evaluate_sb3_model(model_path, model_type, num_episodes=20)\n        else:\n            print(f\"Warning: {model_type.upper()} model not found at {model_path}\")\n    \n    # Evaluate REINFORCE model\n    reinforce_path = \"models/reinforce/reinforce_cattle_final.pth\"\n    if os.path.exists(reinforce_path):\n        print(\"\\nEvaluating REINFORCE...\")\n        evaluator.evaluate_reinforce_model(reinforce_path, num_episodes=20)\n    else:\n        print(f\"Warning: REINFORCE model not found at {reinforce_path}\")\n    \n    # Generate comparison\n    comparison_df = evaluator.compare_algorithms()\n    \n    # Create visualizations\n    evaluator.plot_comparison(\"algorithm_comparison.png\")\n    \n    # Save results\n    evaluator.save_results(\"evaluation_results.json\")\n    \n    print(\"\\nEvaluation completed successfully!\")\n    print(\"Files generated:\")\n    print(\"  - algorithm_comparison.png\")\n    print(\"  - evaluation_results.json\")\n\nif __name__ == \"__main__\":\n    main()