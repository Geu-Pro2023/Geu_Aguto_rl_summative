"""
Random Agent Demonstration for Cattle Verification Environment
This script shows the agent taking random actions without any training.
"""

import numpy as np
import pygame
import time
from environment.custom_env import CattleVerificationEnv
import imageio
import os

def run_random_agent_demo(episodes=3, max_steps=200, save_gif=True):
    """
    Demonstrate random agent behavior in the cattle verification environment
    """
    print("=" * 60)
    print("CATTLE VERIFICATION SYSTEM - RANDOM AGENT DEMONSTRATION")
    print("=" * 60)
    print("This demonstrates the environment visualization without any training.")
    print("The agent (blue circle) moves randomly around the grid.")
    print("Goal: Verify cattle (brown circles) and reach the verification station (green square)")
    print("Avoid: Thieves (red triangles)")
    print("=" * 60)
    
    # Create environment with human rendering
    env = CattleVerificationEnv(render_mode="human", size=10)
    
    # Create recordings directory if saving GIFs
    if save_gif:
        os.makedirs("recordings", exist_ok=True)
    
    try:
        for episode in range(1, episodes + 1):
            print(f"\nStarting Episode {episode}/{episodes}")
            print("Press Ctrl+C to stop early")
            
            # Reset environment
            obs, info = env.reset()
            total_reward = 0
            frames = []
            
            print(f"Initial state:")
            print(f"  Agent position: {obs['agent']}")
            print(f"  Target position: {obs['target']}")
            print(f"  Cattle positions: {obs['cattle']}")
            print(f"  Thief positions: {obs['thieves']}")
            
            for step in range(max_steps):
                # Take random action
                action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # Render environment
                env.render()
                
                # Save frame for GIF if requested
                if save_gif:
                    frame = env._render_frame()
                    if frame is not None:
                        frames.append(frame)
                
                # Print step info
                action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
                print(f"Step {step + 1:3d}: Action={action_names[action]:5s} | "
                      f"Reward={reward:6.2f} | Total={total_reward:7.2f} | "
                      f"Verified={info['verified_count']}/4", end="\r")
                
                # Add small delay for better visualization
                time.sleep(0.1)
                
                if terminated or truncated:
                    print(f"\nEpisode {episode} terminated after {step + 1} steps")
                    break
            
            # Episode summary
            print(f"\nEpisode {episode} Summary:")
            print(f"  Total Steps: {step + 1}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Cattle Verified: {info['verified_count']}/4")
            print(f"  Distance to Target: {info['distance_to_target']:.1f}")
            print(f"  Thief Proximity: {info['thief_proximity']:.1f}")
            
            # Save GIF for this episode
            if save_gif and frames:
                gif_path = f"recordings/random_agent_episode_{episode}.gif"
                imageio.mimsave(gif_path, frames, fps=8)
                print(f"  Saved GIF: {gif_path}")
            
            # Wait between episodes
            if episode < episodes:
                print("\nPress Enter to continue to next episode...")
                input()
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    
    finally:
        env.close()
        print("\nRandom agent demonstration completed!")

def analyze_random_performance(num_runs=10, max_steps=200):
    """
    Analyze random agent performance over multiple runs
    """
    print("\n" + "=" * 50)
    print("RANDOM AGENT PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    env = CattleVerificationEnv(render_mode=None, size=10)
    
    rewards = []
    steps_taken = []
    cattle_verified = []
    success_rate = 0
    
    for run in range(num_runs):
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                if total_reward > 0:  # Successful episode
                    success_rate += 1
                break
        
        rewards.append(total_reward)
        steps_taken.append(step + 1)
        cattle_verified.append(info['verified_count'])
        
        print(f"Run {run + 1:2d}: Reward={total_reward:7.2f}, "
              f"Steps={step + 1:3d}, Verified={info['verified_count']}/4")
    
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_steps = np.mean(steps_taken)
    avg_verified = np.mean(cattle_verified)
    success_rate = success_rate / num_runs * 100
    
    print("\n" + "-" * 50)
    print("RANDOM AGENT STATISTICS:")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"  Average Cattle Verified: {avg_verified:.1f}/4")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Best Reward: {max(rewards):.2f}")
    print(f"  Worst Reward: {min(rewards):.2f}")
    print("-" * 50)

if __name__ == "__main__":
    print("Cattle Verification System - Random Agent Demo")
    print("This script demonstrates the environment without any RL training.")
    
    # Run visual demonstration
    run_random_agent_demo(episodes=3, max_steps=150, save_gif=True)
    
    # Run performance analysis
    analyze_random_performance(num_runs=20, max_steps=200)