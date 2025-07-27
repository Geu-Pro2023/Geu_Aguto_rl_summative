#!/usr/bin/env python3
"""
Main entry point for Cattle Monitoring RL Assignment
Geu Aguto Garang Bior - BSE ML Techniques II Summative Assignment
"""

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add environment to path
sys.path.append('environment')
sys.path.append('training')

from environment.custom_env import CattleMonitoringEnv
from environment.rendering import create_demo_gif

try:
    from training.dqn_training import DQNTrainer
    from training.pg_training import PolicyGradientTrainer
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available. Using simplified training.")
    SB3_AVAILABLE = False

class ExperimentRunner:
    """Main experiment runner for RL cattle monitoring"""
    
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_static_demo(self):
        """Run static demonstration with random actions"""
        print("="*60)
        print("RUNNING STATIC DEMONSTRATION")
        print("="*60)
        
        try:
            create_demo_gif()
            print("✓ Static demonstration completed successfully")
        except Exception as e:
            print(f"✗ Error in static demonstration: {e}")
            
    def train_all_models(self, timesteps=50000):
        """Train all RL models"""
        if not SB3_AVAILABLE:
            print("Running simplified training without stable-baselines3...")
            import subprocess
            subprocess.run(["python", "simple_training.py"])
            return
            
        print("="*60)
        print("TRAINING ALL RL MODELS")
        print("="*60)
        
        # Train DQN
        print("\n1. Training DQN (Value-Based Method)")
        print("-" * 40)
        try:
            dqn_trainer = DQNTrainer()
            dqn_model = dqn_trainer.train_model(total_timesteps=timesteps)
            self.results['dqn'] = dqn_trainer.evaluate_model("models/dqn/dqn_final")
            print("✓ DQN training completed")
        except Exception as e:
            print(f"✗ DQN training failed: {e}")
            
        # Train Policy Gradient Methods
        print("\n2. Training Policy Gradient Methods")
        print("-" * 40)
        try:
            pg_trainer = PolicyGradientTrainer()
            
            # PPO
            print("Training PPO...")
            ppo_model = pg_trainer.train_ppo(total_timesteps=timesteps)
            self.results['ppo'] = pg_trainer.evaluate_model("models/pg/ppo_final", "ppo")
            
            # A2C (Actor-Critic)
            print("Training A2C (Actor-Critic)...")
            a2c_model = pg_trainer.train_a2c(total_timesteps=timesteps)
            self.results['a2c'] = pg_trainer.evaluate_model("models/pg/a2c_final", "a2c")
            
            # REINFORCE
            print("Training REINFORCE...")
            reinforce_agent, _ = pg_trainer.train_reinforce(episodes=500)
            self.results['reinforce'] = pg_trainer.evaluate_model(
                "models/pg/reinforce/reinforce_final.pth", "reinforce"
            )
            
            print("✓ All Policy Gradient methods completed")
        except Exception as e:
            print(f"✗ Policy Gradient training failed: {e}")
            
    def compare_results(self):
        """Compare and visualize results"""
        print("="*60)
        print("COMPARING RESULTS")
        print("="*60)
        
        if not self.results:
            print("No results to compare. Run training first.")
            return
            
        # Print comparison table
        print(f"{'Algorithm':<12} {'Mean Reward':<12} {'Success Rate':<12} {'Avg Length':<12}")
        print("-" * 50)
        
        for algo, result in self.results.items():
            print(f"{algo.upper():<12} {result['mean_reward']:<12.2f} "
                  f"{result['success_rate']:<12.1f}% {result['mean_length']:<12.1f}")
                  
        # Create visualization
        self._create_comparison_plots()
        
        # Save results
        self._save_results()
        
    def _create_comparison_plots(self):
        """Create comparison plots"""
        algorithms = list(self.results.keys())
        mean_rewards = [self.results[algo]['mean_reward'] for algo in algorithms]
        success_rates = [self.results[algo]['success_rate'] for algo in algorithms]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mean rewards comparison
        bars1 = ax1.bar(algorithms, mean_rewards, color=['blue', 'green', 'red', 'orange'])
        ax1.set_title('Mean Reward Comparison')
        ax1.set_ylabel('Mean Reward')
        ax1.set_xlabel('Algorithm')
        
        # Add value labels on bars
        for bar, reward in zip(bars1, mean_rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # Success rate comparison
        bars2 = ax2.bar(algorithms, success_rates, color=['blue', 'green', 'red', 'orange'])
        ax2.set_title('Success Rate Comparison')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars2, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'results_comparison_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Comparison plots saved as 'results_comparison_{self.timestamp}.png'")
        
    def _save_results(self):
        """Save results to JSON file"""
        filename = f'experiment_results_{self.timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to '{filename}'")
        
    def demonstrate_best_model(self):
        """Demonstrate the best performing model"""
        if not SB3_AVAILABLE:
            print("Demonstrating with simplified random policy...")
            env = CattleMonitoringEnv()
            for episode in range(3):
                print(f"\nEpisode {episode + 1}")
                obs, info = env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                while not done and steps < 100:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated
                    
                print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}")
                print(f"Cattle Registered: {info['cattle_registered']}, Alerts: {info['alerts_sent']}")
            return
            
        if not self.results:
            print("No results available. Run training first.")
            return
            
        # Find best model by mean reward
        best_algo = max(self.results.keys(), key=lambda x: self.results[x]['mean_reward'])
        print(f"\nDemonstrating best model: {best_algo.upper()}")
        print(f"Mean Reward: {self.results[best_algo]['mean_reward']:.2f}")
        print(f"Success Rate: {self.results[best_algo]['success_rate']:.1f}%")
        
        # Load and demonstrate best model
        env = CattleMonitoringEnv(render_mode="human")
        
        if best_algo == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load("models/dqn/dqn_final")
        elif best_algo == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load("models/pg/ppo_final")
        elif best_algo == "a2c":
            from stable_baselines3 import A2C
            model = A2C.load("models/pg/a2c_final")
        else:  # reinforce
            import torch
            from training.pg_training import REINFORCEAgent
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            model = REINFORCEAgent(state_size, action_size)
            model.policy_net.load_state_dict(torch.load("models/pg/reinforce/reinforce_final.pth"))
            model.policy_net.eval()
        
        # Run 3 episodes
        for episode in range(3):
            print(f"\nEpisode {episode + 1}")
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 200:
                if best_algo == "reinforce":
                    action = model.select_action(obs)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
                
            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}")
            print(f"Cattle Registered: {info['cattle_registered']}, Alerts: {info['alerts_sent']}")
            
        env.close()

def main():
    parser = argparse.ArgumentParser(description='Cattle Monitoring RL Assignment')
    parser.add_argument('--demo', action='store_true', help='Run static demonstration')
    parser.add_argument('--train', action='store_true', help='Train all models')
    parser.add_argument('--compare', action='store_true', help='Compare results')
    parser.add_argument('--best', action='store_true', help='Demonstrate best model')
    parser.add_argument('--all', action='store_true', help='Run complete experiment')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.all or args.demo:
        runner.run_static_demo()
        
    if args.all or args.train:
        runner.train_all_models(timesteps=args.timesteps)
        
    if args.all or args.compare:
        runner.compare_results()
        
    if args.all or args.best:
        runner.demonstrate_best_model()
        
    if not any([args.demo, args.train, args.compare, args.best, args.all]):
        print("Cattle Monitoring RL Assignment")
        print("Usage: python main.py [--demo] [--train] [--compare] [--best] [--all]")
        print("\nOptions:")
        print("  --demo     Run static demonstration with random actions")
        print("  --train    Train all RL models (DQN, PPO, A2C, REINFORCE)")
        print("  --compare  Compare and visualize results")
        print("  --best     Demonstrate best performing model")
        print("  --all      Run complete experiment pipeline")
        print("  --timesteps N  Set training timesteps (default: 50000)")

if __name__ == "__main__":
    main()