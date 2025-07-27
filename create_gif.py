#!/usr/bin/env python3
"""
Create GIF of agent in simulated environment
"""

import sys
import pygame
import numpy as np
from PIL import Image
import os

sys.path.append('environment')
from environment.custom_env import CattleMonitoringEnv

def create_agent_gif():
    """Create GIF showing agent taking random actions"""
    
    # Initialize environment with pygame rendering
    env = CattleMonitoringEnv(render_mode="rgb_array")
    
    frames = []
    max_frames = 100  # Limit for GIF size
    
    print("Creating GIF of agent demonstration...")
    
    for episode in range(3):  # 3 episodes as requested
        print(f"Recording episode {episode + 1}...")
        
        obs, info = env.reset()
        episode_frames = 0
        
        while episode_frames < 30 and len(frames) < max_frames:  # 30 frames per episode max
            # Render current state
            frame = env.render()
            if frame is not None:
                # Convert to PIL Image
                img = Image.fromarray(frame)
                frames.append(img)
            
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_frames += 1
            
            if terminated or truncated:
                break
    
    env.close()
    
    if frames:
        # Save as GIF
        frames[0].save(
            'cattle_monitoring_demo.gif',
            save_all=True,
            append_images=frames[1:],
            duration=500,  # 500ms per frame
            loop=0
        )
        print(f"✓ GIF created: cattle_monitoring_demo.gif ({len(frames)} frames)")
    else:
        print("✗ No frames captured for GIF")

def create_static_demo_images():
    """Create static images showing different states"""
    
    env = CattleMonitoringEnv(render_mode="rgb_array")
    
    images = []
    descriptions = []
    
    # Capture different scenarios
    scenarios = [
        "Initial state - Agent starts mission",
        "Agent moving towards cattle",
        "Agent at registration point",
        "Agent avoiding raiders",
        "Agent at checkpoint for alert"
    ]
    
    for i, description in enumerate(scenarios):
        obs, info = env.reset()
        
        # Take some random actions to get different states
        for _ in range(i * 5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        # Capture frame
        frame = env.render()
        if frame is not None:
            img = Image.fromarray(frame)
            img.save(f'demo_state_{i+1}.png')
            images.append(img)
            descriptions.append(description)
            print(f"✓ Saved: demo_state_{i+1}.png - {description}")
    
    env.close()
    
    # Create a combined image with descriptions
    if images:
        # Calculate combined image size
        width = max(img.width for img in images)
        height = sum(img.height + 30 for img in images)  # 30px for text
        
        combined = Image.new('RGB', (width, height), 'white')
        
        y_offset = 0
        for img, desc in zip(images, descriptions):
            combined.paste(img, (0, y_offset))
            y_offset += img.height + 30
        
        combined.save('cattle_monitoring_states.png')
        print("✓ Combined image saved: cattle_monitoring_states.png")

if __name__ == "__main__":
    print("Creating demonstration materials...")
    
    try:
        create_agent_gif()
        create_static_demo_images()
        print("\\n✓ All demonstration materials created successfully!")
    except Exception as e:
        print(f"Error creating demonstrations: {e}")
        print("Make sure pygame is properly installed and display is available.")