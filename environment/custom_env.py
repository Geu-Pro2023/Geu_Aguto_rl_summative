import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import pygame
PYGAME_AVAILABLE = True

class CattleMonitoringEnv(gym.Env):
    """
    Custom Environment for Cattle Monitoring and Theft Prevention in South Sudan
    
    The agent (herder) navigates a 10x10 grid representing rural landscape to:
    - Register cattle at registration points
    - Detect stolen cattle
    - Alert authorities at checkpoints
    - Avoid raiders and dangerous areas
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, grid_size=10):
        self.grid_size = grid_size
        self.window_size = 512
        
        # Observation space: agent position + cattle positions + raider positions + checkpoint status
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, 
            shape=(2 + 6 + 4 + 3,), dtype=np.int32  # agent(2) + cattle(6) + raiders(4) + checkpoints(3)
        )
        
        # Action space: 6 discrete actions
        # 0: Move Up, 1: Move Down, 2: Move Left, 3: Move Right, 4: Register Cattle, 5: Alert Authorities
        self.action_space = spaces.Discrete(6)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Environment entities
        self.agent_pos = None
        self.cattle_positions = []
        self.raider_positions = []
        self.registration_points = []
        self.checkpoints = []
        self.stolen_cattle = set()
        
        # Rewards
        self.cattle_registered = 0
        self.alerts_sent = 0
        self.step_count = 0
        self.max_steps = 200
        
    def _get_obs(self):
        """Get current observation"""
        obs = np.zeros(15, dtype=np.int32)
        
        # Agent position
        obs[0:2] = self.agent_pos
        
        # Cattle positions (3 cattle, 2 coords each)
        for i, cattle_pos in enumerate(self.cattle_positions[:3]):
            obs[2 + i*2:4 + i*2] = cattle_pos
            
        # Raider positions (2 raiders, 2 coords each)  
        for i, raider_pos in enumerate(self.raider_positions[:2]):
            obs[8 + i*2:10 + i*2] = raider_pos
            
        # Checkpoint status (3 checkpoints)
        for i, checkpoint in enumerate(self.checkpoints[:3]):
            obs[12 + i] = 1 if np.array_equal(self.agent_pos, checkpoint) else 0
            
        return obs
    
    def _get_info(self):
        """Get additional info"""
        return {
            "cattle_registered": self.cattle_registered,
            "alerts_sent": self.alerts_sent,
            "stolen_cattle_detected": len(self.stolen_cattle),
            "step_count": self.step_count
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent position
        self.agent_pos = np.array([0, 0])
        
        # Place cattle randomly
        self.cattle_positions = []
        for _ in range(3):
            pos = self._get_random_position()
            self.cattle_positions.append(pos)
            
        # Place raiders randomly
        self.raider_positions = []
        for _ in range(2):
            pos = self._get_random_position()
            self.raider_positions.append(pos)
            
        # Place registration points
        self.registration_points = [
            np.array([2, 8]), np.array([8, 2]), np.array([5, 5])
        ]
        
        # Place checkpoints
        self.checkpoints = [
            np.array([1, 9]), np.array([9, 1]), np.array([9, 9])
        ]
        
        # Reset counters
        self.cattle_registered = 0
        self.alerts_sent = 0
        self.step_count = 0
        self.stolen_cattle = set()
        
        # Randomly mark some cattle as stolen
        if random.random() < 0.3:
            self.stolen_cattle.add(0)
        if random.random() < 0.2:
            self.stolen_cattle.add(1)
            
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action):
        self.step_count += 1
        reward = -0.1  # Small negative reward for each step
        terminated = False
        
        # Execute action
        if action < 4:  # Movement actions
            self._move_agent(action)
        elif action == 4:  # Register cattle
            reward += self._register_cattle()
        elif action == 5:  # Alert authorities
            reward += self._alert_authorities()
            
        # Move raiders randomly
        self._move_raiders()
        
        # Check for collisions with raiders
        for raider_pos in self.raider_positions:
            if np.array_equal(self.agent_pos, raider_pos):
                reward -= 10  # Heavy penalty for encountering raiders
                
        # Check win condition
        if self.cattle_registered >= 2 and self.alerts_sent >= 1:
            reward += 50
            terminated = True
            
        # Check lose condition
        if self.step_count >= self.max_steps:
            reward -= 20
            terminated = True
            
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, False, info
    
    def _move_agent(self, action):
        """Move agent based on action"""
        if action == 0 and self.agent_pos[1] < self.grid_size - 1:  # Up
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0:  # Down
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0:  # Left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # Right
            self.agent_pos[0] += 1
            
    def _register_cattle(self):
        """Register cattle at registration points"""
        reward = 0
        for reg_point in self.registration_points:
            if np.array_equal(self.agent_pos, reg_point):
                # Check if there's cattle nearby
                for i, cattle_pos in enumerate(self.cattle_positions[:]):
                    if np.linalg.norm(self.agent_pos - cattle_pos) <= 1.5:
                        self.cattle_registered += 1
                        reward += 15
                        # Remove registered cattle
                        self.cattle_positions = [cp for cp in self.cattle_positions if not np.array_equal(cp, cattle_pos)]
                        break
        return reward
    
    def _alert_authorities(self):
        """Alert authorities at checkpoints"""
        reward = 0
        for checkpoint in self.checkpoints:
            if np.array_equal(self.agent_pos, checkpoint):
                # Check if stolen cattle detected
                if len(self.stolen_cattle) > 0:
                    self.alerts_sent += 1
                    reward += 20
                    self.stolen_cattle.clear()
                else:
                    reward -= 2  # Small penalty for false alert
        return reward
    
    def _move_raiders(self):
        """Move raiders randomly"""
        for i, raider_pos in enumerate(self.raider_positions):
            action = random.randint(0, 3)
            if action == 0 and raider_pos[1] < self.grid_size - 1:
                raider_pos[1] += 1
            elif action == 1 and raider_pos[1] > 0:
                raider_pos[1] -= 1
            elif action == 2 and raider_pos[0] > 0:
                raider_pos[0] -= 1
            elif action == 3 and raider_pos[0] < self.grid_size - 1:
                raider_pos[0] += 1
                
    def _get_random_position(self):
        """Get random position on grid"""
        return np.array([
            random.randint(1, self.grid_size-2),
            random.randint(1, self.grid_size-2)
        ])
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
    
    def _render_frame(self):
        if not PYGAME_AVAILABLE:
            # Text-based rendering fallback
            self._render_text()
            return None
            
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((34, 139, 34))  # Forest green background
        
        pix_square_size = self.window_size / self.grid_size
        
        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, (0, 0, 0),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas, (0, 0, 0),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )
        
        # Draw registration points (blue squares)
        for reg_point in self.registration_points:
            pygame.draw.rect(
                canvas, (0, 0, 255),
                pygame.Rect(
                    pix_square_size * reg_point[0],
                    pix_square_size * reg_point[1],
                    pix_square_size, pix_square_size
                ),
            )
            
        # Draw checkpoints (yellow squares)
        for checkpoint in self.checkpoints:
            pygame.draw.rect(
                canvas, (255, 255, 0),
                pygame.Rect(
                    pix_square_size * checkpoint[0],
                    pix_square_size * checkpoint[1],
                    pix_square_size, pix_square_size
                ),
            )
        
        # Draw cattle (brown circles)
        for i, cattle_pos in enumerate(self.cattle_positions):
            color = (139, 69, 19) if i not in self.stolen_cattle else (255, 0, 0)
            pygame.draw.circle(
                canvas, color,
                (cattle_pos * pix_square_size + pix_square_size / 2).astype(int),
                pix_square_size // 3,
            )
            
        # Draw raiders (red triangles)
        for raider_pos in self.raider_positions:
            center = (raider_pos * pix_square_size + pix_square_size / 2).astype(int)
            points = [
                (center[0], center[1] - pix_square_size // 3),
                (center[0] - pix_square_size // 3, center[1] + pix_square_size // 3),
                (center[0] + pix_square_size // 3, center[1] + pix_square_size // 3)
            ]
            pygame.draw.polygon(canvas, (255, 0, 0), points)
        
        # Draw agent (green circle)
        pygame.draw.circle(
            canvas, (0, 255, 0),
            (self.agent_pos * pix_square_size + pix_square_size / 2).astype(int),
            pix_square_size // 3,
        )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _render_text(self):
        """Text-based rendering fallback"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place entities
        for reg_point in self.registration_points:
            grid[reg_point[1]][reg_point[0]] = 'R'
        for checkpoint in self.checkpoints:
            grid[checkpoint[1]][checkpoint[0]] = 'C'
        for i, cattle_pos in enumerate(self.cattle_positions):
            symbol = 'S' if i in self.stolen_cattle else 'B'
            grid[cattle_pos[1]][cattle_pos[0]] = symbol
        for raider_pos in self.raider_positions:
            grid[raider_pos[1]][raider_pos[0]] = 'X'
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A'
        
        # Print grid
        print("\n" + "="*30)
        for row in reversed(grid):
            print(' '.join(row))
        print("A=Agent, B=Cattle, S=Stolen, X=Raider, R=Registration, C=Checkpoint")
        print(f"Registered: {self.cattle_registered}, Alerts: {self.alerts_sent}")
    
    def close(self):
        if PYGAME_AVAILABLE and self.window is not None:
            pygame.display.quit()
            pygame.quit()