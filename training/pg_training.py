import os
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
sys.path.append('../environment')
from custom_env import CattleMonitoringEnv

class REINFORCEAgent:
    """REINFORCE Algorithm Implementation"""
    
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update_policy(self, gamma=0.99):
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate returns
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        del self.rewards[:]
        del self.saved_log_probs[:]

class PolicyGradientTrainer:
    """Training class for Policy Gradient methods"""
    
    def __init__(self, model_save_path="../models/pg/"):
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(f"{model_save_path}/reinforce", exist_ok=True)
        os.makedirs(f"{model_save_path}/actor_critic", exist_ok=True)
        
        # PPO Hyperparameters
        self.ppo_hyperparams = {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'tensorboard_log': "./ppo_cattle_tensorboard/"
        }
        
        # A2C Hyperparameters  
        self.a2c_hyperparams = {
            'learning_rate': 0.0007,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.25,
            'max_grad_norm': 0.5,
            'tensorboard_log': "./a2c_cattle_tensorboard/"
        }
        
    def create_environment(self):
        """Create and wrap environment"""
        def make_env():
            env = CattleMonitoringEnv()
            env = Monitor(env)
            return env
        
        return make_vec_env(make_env, n_envs=1)
    
    def train_ppo(self, total_timesteps=100000):
        """Train PPO model"""
        print("Training PPO...")
        env = self.create_environment()
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            **self.ppo_hyperparams
        )
        
        eval_env = self.create_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.model_save_path}/ppo/",
            log_path=f"{self.model_save_path}/ppo/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        model.save(f"{self.model_save_path}/ppo_final")
        return model
    
    def train_a2c(self, total_timesteps=100000):
        """Train A2C (Actor-Critic) model"""
        print("Training A2C (Actor-Critic)...")
        env = self.create_environment()
        
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            **self.a2c_hyperparams
        )
        
        eval_env = self.create_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.model_save_path}/actor_critic/",
            log_path=f"{self.model_save_path}/actor_critic/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        model.save(f"{self.model_save_path}/a2c_final")
        return model
    
    def train_reinforce(self, episodes=1000):
        """Train REINFORCE model"""
        print("Training REINFORCE...")
        env = CattleMonitoringEnv()
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = REINFORCEAgent(state_size, action_size)
        
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                action = agent.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                agent.rewards.append(reward)
                episode_reward += reward
                
                if terminated or truncated:
                    break
                    
            agent.update_policy()
            episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
                
        # Save REINFORCE model
        torch.save(agent.policy_net.state_dict(), 
                  f"{self.model_save_path}/reinforce/reinforce_final.pth")
        
        return agent, episode_rewards
    
    def evaluate_model(self, model_path, algorithm, n_episodes=10):
        """Evaluate trained model"""
        env = CattleMonitoringEnv()
        
        if algorithm == "reinforce":
            # Load REINFORCE model
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            agent = REINFORCEAgent(state_size, action_size)
            agent.policy_net.load_state_dict(torch.load(model_path))
            agent.policy_net.eval()
        else:
            # Load SB3 model
            if algorithm == "ppo":
                model = PPO.load(model_path)
            elif algorithm == "a2c":
                model = A2C.load(model_path)
        
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                if algorithm == "reinforce":
                    action = agent.select_action(obs)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if info['cattle_registered'] >= 2 and info['alerts_sent'] >= 1:
                success_rate += 1
                
        success_rate = success_rate / n_episodes * 100
        
        results = {
            'algorithm': algorithm.upper(),
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_rate,
            'episode_rewards': episode_rewards
        }
        
        print(f"\n{algorithm.upper()} Evaluation Results ({n_episodes} episodes):")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Episode Length: {results['mean_length']:.2f}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        
        return results

def main():
    trainer = PolicyGradientTrainer()
    
    # Train all models
    print("Training Policy Gradient Methods...")
    
    # Train PPO
    ppo_model = trainer.train_ppo(total_timesteps=50000)
    
    # Train A2C (Actor-Critic)
    a2c_model = trainer.train_a2c(total_timesteps=50000)
    
    # Train REINFORCE
    reinforce_agent, reinforce_rewards = trainer.train_reinforce(episodes=500)
    
    # Evaluate all models
    results = {}
    
    results['ppo'] = trainer.evaluate_model("../models/pg/ppo_final", "ppo")
    results['a2c'] = trainer.evaluate_model("../models/pg/a2c_final", "a2c")
    results['reinforce'] = trainer.evaluate_model(
        "../models/pg/reinforce/reinforce_final.pth", "reinforce"
    )
    
    return results

if __name__ == "__main__":
    main()