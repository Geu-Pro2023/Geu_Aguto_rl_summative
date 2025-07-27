import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import sys
sys.path.append('../environment')
from custom_env import CattleMonitoringEnv

class DQNTrainer:
    """DQN Training for Cattle Monitoring Environment"""
    
    def __init__(self, model_save_path="../models/dqn/"):
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        
        # Hyperparameters for DQN
        self.hyperparams = {
            'learning_rate': 0.0001,
            'buffer_size': 50000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'max_grad_norm': 10,
            'tensorboard_log': "./dqn_cattle_tensorboard/"
        }
        
    def create_environment(self):
        """Create and wrap environment"""
        def make_env():
            env = CattleMonitoringEnv()
            env = Monitor(env)
            return env
        
        return make_vec_env(make_env, n_envs=1)
    
    def train_model(self, total_timesteps=100000):
        """Train DQN model"""
        print("Creating environment...")
        env = self.create_environment()
        
        print("Initializing DQN model...")
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            **self.hyperparams
        )
        
        # Callback for evaluation
        eval_env = self.create_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_save_path,
            log_path=self.model_save_path,
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        print(f"Training DQN for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final model
        model.save(os.path.join(self.model_save_path, "dqn_final"))
        print(f"Model saved to {self.model_save_path}")
        
        return model
    
    def evaluate_model(self, model_path, n_episodes=10):
        """Evaluate trained model"""
        env = CattleMonitoringEnv()
        model = DQN.load(model_path)
        
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check success (cattle registered >= 2 and alerts sent >= 1)
            if info['cattle_registered'] >= 2 and info['alerts_sent'] >= 1:
                success_rate += 1
                
        success_rate = success_rate / n_episodes * 100
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_rate,
            'episode_rewards': episode_rewards
        }
        
        print(f"\nDQN Evaluation Results ({n_episodes} episodes):")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Episode Length: {results['mean_length']:.2f}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        
        return results

def main():
    trainer = DQNTrainer()
    
    # Train model
    model = trainer.train_model(total_timesteps=50000)
    
    # Evaluate model
    model_path = "../models/dqn/dqn_final"
    results = trainer.evaluate_model(model_path)
    
    return results

if __name__ == "__main__":
    main()