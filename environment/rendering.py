import numpy as np
import random
import time
from custom_env import CattleMonitoringEnv

import pygame
PYGAME_AVAILABLE = True

class CattleVisualization:
    """Advanced visualization for cattle monitoring environment"""
    
    def __init__(self, env):
        self.env = env
        if PYGAME_AVAILABLE:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("Cattle Monitoring - South Sudan")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        else:
            self.screen = None
            self.clock = None
            self.font = None
        
    def render_static_demo(self, episodes=3):
        """Render agent taking random actions for demonstration"""
        print("Starting static demonstration with random actions...")
        
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}")
            obs, info = self.env.reset()
            done = False
            step = 0
            total_reward = 0
            
            while not done and step < 50:  # Reduced steps for text demo
                # Take random action
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1
                
                # Render current state
                if PYGAME_AVAILABLE:
                    self._render_advanced_frame(info, step, total_reward, episode + 1)
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                else:
                    self._render_text_frame(info, step, total_reward, episode + 1, action)
                        
                time.sleep(0.5)  # Slow down for visibility
                
            print(f"Episode {episode + 1} completed: Steps={step}, Reward={total_reward:.2f}")
            print(f"Final Status: Registered={info['cattle_registered']}, Alerts={info['alerts_sent']}")
            time.sleep(1)
            
        if PYGAME_AVAILABLE:
            pygame.quit()
        
    def _render_advanced_frame(self, info, step, total_reward, episode):
        """Render advanced frame with detailed information"""
        if not PYGAME_AVAILABLE:
            return
            
        self.screen.fill((34, 139, 34))  # Forest green
        
        grid_size = self.env.grid_size
        cell_size = 500 // grid_size
        offset_x, offset_y = 50, 50
        
        # Draw grid
        for i in range(grid_size + 1):
            pygame.draw.line(self.screen, (0, 0, 0),
                           (offset_x, offset_y + i * cell_size),
                           (offset_x + grid_size * cell_size, offset_y + i * cell_size))
            pygame.draw.line(self.screen, (0, 0, 0),
                           (offset_x + i * cell_size, offset_y),
                           (offset_x + i * cell_size, offset_y + grid_size * cell_size))
        
        # Draw registration points (blue)
        for reg_point in self.env.registration_points:
            x = offset_x + reg_point[0] * cell_size
            y = offset_y + reg_point[1] * cell_size
            pygame.draw.rect(self.screen, (0, 100, 255),
                           (x + 2, y + 2, cell_size - 4, cell_size - 4))
            
        # Draw checkpoints (yellow)
        for checkpoint in self.env.checkpoints:
            x = offset_x + checkpoint[0] * cell_size
            y = offset_y + checkpoint[1] * cell_size
            pygame.draw.rect(self.screen, (255, 255, 0),
                           (x + 2, y + 2, cell_size - 4, cell_size - 4))
        
        # Draw cattle
        for i, cattle_pos in enumerate(self.env.cattle_positions):
            x = offset_x + cattle_pos[0] * cell_size + cell_size // 2
            y = offset_y + cattle_pos[1] * cell_size + cell_size // 2
            color = (139, 69, 19) if i not in self.env.stolen_cattle else (255, 0, 0)
            pygame.draw.circle(self.screen, color, (x, y), cell_size // 3)
            
        # Draw raiders
        for raider_pos in self.env.raider_positions:
            x = offset_x + raider_pos[0] * cell_size + cell_size // 2
            y = offset_y + raider_pos[1] * cell_size + cell_size // 2
            points = [
                (x, y - cell_size // 3),
                (x - cell_size // 3, y + cell_size // 3),
                (x + cell_size // 3, y + cell_size // 3)
            ]
            pygame.draw.polygon(self.screen, (255, 0, 0), points)
            
        # Draw agent (herder)
        agent_x = offset_x + self.env.agent_pos[0] * cell_size + cell_size // 2
        agent_y = offset_y + self.env.agent_pos[1] * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, (0, 255, 0), (agent_x, agent_y), cell_size // 3)
        
        # Draw information panel
        info_y = 20
        texts = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Total Reward: {total_reward:.2f}",
            f"Cattle Registered: {info['cattle_registered']}",
            f"Alerts Sent: {info['alerts_sent']}",
            f"Stolen Detected: {info['stolen_cattle_detected']}"
        ]
        
        for i, text in enumerate(texts):
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, info_y + i * 25))
            
        # Draw legend
        legend_x = 400
        legend_items = [
            ("Agent (Herder)", (0, 255, 0)),
            ("Cattle", (139, 69, 19)),
            ("Stolen Cattle", (255, 0, 0)),
            ("Raiders", (255, 0, 0)),
            ("Registration", (0, 100, 255)),
            ("Checkpoint", (255, 255, 0))
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = 400 + i * 25
            pygame.draw.circle(self.screen, color, (legend_x, y_pos), 8)
            surface = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(surface, (legend_x + 20, y_pos - 8))
        
        pygame.display.flip()
        self.clock.tick(5)
    
    def _render_text_frame(self, info, step, total_reward, episode, action):
        """Text-based rendering for when pygame is not available"""
        action_names = ['Up', 'Down', 'Left', 'Right', 'Register', 'Alert']
        
        print(f"\n--- Episode {episode}, Step {step} ---")
        print(f"Action: {action_names[action]} ({action})")
        print(f"Agent Position: ({self.env.agent_pos[0]}, {self.env.agent_pos[1]})")
        print(f"Reward: {total_reward:.2f}")
        print(f"Cattle Registered: {info['cattle_registered']}")
        print(f"Alerts Sent: {info['alerts_sent']}")
        print(f"Stolen Detected: {info['stolen_cattle_detected']}")
        
        # Simple grid representation
        grid = [['.' for _ in range(self.env.grid_size)] for _ in range(self.env.grid_size)]
        
        # Place entities
        for reg_point in self.env.registration_points:
            grid[reg_point[1]][reg_point[0]] = 'R'
        for checkpoint in self.env.checkpoints:
            grid[checkpoint[1]][checkpoint[0]] = 'C'
        for i, cattle_pos in enumerate(self.env.cattle_positions):
            symbol = 'S' if i in self.env.stolen_cattle else 'B'
            grid[cattle_pos[1]][cattle_pos[0]] = symbol
        for raider_pos in self.env.raider_positions:
            grid[raider_pos[1]][raider_pos[0]] = 'X'
        grid[self.env.agent_pos[1]][self.env.agent_pos[0]] = 'A'
        
        # Print grid (flipped for correct orientation)
        for row in reversed(grid):
            print(' '.join(row))
        print("Legend: A=Agent, B=Cattle, S=Stolen, X=Raider, R=Registration, C=Checkpoint")

def create_demo_gif():
    """Create a static demonstration of the environment"""
    render_mode = "human" if PYGAME_AVAILABLE else None
    env = CattleMonitoringEnv(render_mode=render_mode)
    viz = CattleVisualization(env)
    viz.render_static_demo(episodes=3)

if __name__ == "__main__":
    create_demo_gif()