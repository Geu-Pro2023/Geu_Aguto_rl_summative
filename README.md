# Cattle Monitoring RL Assignment - Geu Aguto Garang Bior

## Non-IoT Cattle Raiding Prevention System Using Reinforcement Learning

This project implements a reinforcement learning solution for cattle monitoring and theft prevention in South Sudan, comparing four different RL algorithms: DQN (Value-Based), REINFORCE, PPO, and Actor-Critic methods.

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Advanced visualization GUI components
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO/REINFORCE/A2C using SB3
├── models/
│   ├── dqn/                     # Saved DQN models
│   ├── pg/                      # Saved policy gradient models
│   │   ├── ppo/                 # PPO models
│   │   ├── actor_critic/        # A2C models
│   │   └── reinforce/           # REINFORCE models
├── main.py                      # Entry point for running experiments
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Environment Description

### Cattle Monitoring Environment
- **Grid Size**: 10x10 representing rural South Sudan landscape
- **Agent**: Herder navigating the environment
- **Entities**:
  - 🟢 Agent (Herder)
  - 🟤 Cattle (brown circles, red if stolen)
  - 🔺 Raiders (red triangles)
  - 🟦 Registration Points (blue squares)
  - 🟨 Checkpoints (yellow squares)

### Action Space (Discrete - 6 actions)
0. Move Up
1. Move Down  
2. Move Left
3. Move Right
4. Register Cattle (at registration points)
5. Alert Authorities (at checkpoints)

### Observation Space
- Agent position (2D)
- Cattle positions (6D - 3 cattle × 2 coordinates)
- Raider positions (4D - 2 raiders × 2 coordinates)  
- Checkpoint status (3D - binary for each checkpoint)
- **Total**: 15-dimensional continuous space

### Reward Structure
- **Step penalty**: -0.1 (encourages efficiency)
- **Cattle registration**: +15 (at registration points)
- **Alert authorities**: +20 (when stolen cattle detected)
- **False alert**: -2 (alerting without stolen cattle)
- **Raider encounter**: -10 (collision penalty)
- **Mission success**: +50 (≥2 cattle registered + ≥1 alert sent)
- **Timeout penalty**: -20 (exceeding max steps)

### Termination Conditions
- **Success**: Register ≥2 cattle AND send ≥1 alert
- **Timeout**: Exceed 200 steps
- **Failure**: Continuous poor performance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cattle-monitoring-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Experiment Pipeline
```bash
python main.py --all
```

### Individual Components

#### 1. Static Demonstration (Random Actions)
```bash
python main.py --demo
```
Shows agent taking random actions in the environment with pygame visualization.

#### 2. Train All Models
```bash
python main.py --train --timesteps 50000
```
Trains all four RL algorithms:
- **DQN** (Deep Q-Network)
- **PPO** (Proximal Policy Optimization)
- **A2C** (Actor-Critic)
- **REINFORCE** (Policy Gradient)

#### 3. Compare Results
```bash
python main.py --compare
```
Generates comparison plots and saves results to JSON.

#### 4. Demonstrate Best Model
```bash
python main.py --best
```
Runs 3 episodes with the best performing model.

## Algorithm Implementations

### 1. DQN (Value-Based Method)
- **Network**: MLP with experience replay
- **Key Hyperparameters**:
  - Learning Rate: 0.0001
  - Buffer Size: 50,000
  - Batch Size: 32
  - Target Update: 1000 steps
  - Exploration: ε-greedy (1.0 → 0.05)

### 2. PPO (Policy Gradient)
- **Network**: Actor-Critic with clipped objective
- **Key Hyperparameters**:
  - Learning Rate: 0.0003
  - Clip Range: 0.2
  - GAE Lambda: 0.95
  - Entropy Coefficient: 0.01

### 3. A2C (Actor-Critic)
- **Network**: Shared actor-critic architecture
- **Key Hyperparameters**:
  - Learning Rate: 0.0007
  - N-Steps: 5
  - Value Function Coefficient: 0.25

### 4. REINFORCE (Pure Policy Gradient)
- **Network**: Simple policy network
- **Key Hyperparameters**:
  - Learning Rate: 0.001
  - Gamma: 0.99
  - Baseline: Returns normalization

## Performance Metrics

- **Mean Reward**: Average cumulative reward per episode
- **Success Rate**: Percentage of episodes achieving mission objectives
- **Episode Length**: Average steps to completion
- **Convergence Time**: Training efficiency measure

## Visualization Features

### Advanced Pygame Rendering
- Real-time environment visualization
- Entity tracking and status display
- Performance metrics overlay
- Interactive demonstration mode

### Static Demonstration
- Random action showcase
- Environment component explanation
- Visual legend and information panel

## Expected Results

The system demonstrates:
1. **Environment Complexity**: Multi-objective navigation with dynamic obstacles
2. **Algorithm Comparison**: Performance differences across RL methods
3. **Practical Application**: Real-world cattle monitoring simulation
4. **Hyperparameter Impact**: Tuning effects on learning efficiency

## Technical Specifications

- **Framework**: Gymnasium (OpenAI Gym)
- **RL Library**: Stable-Baselines3
- **Visualization**: Pygame
- **Deep Learning**: PyTorch
- **Python Version**: 3.8+

## Mission Context

This project addresses cattle raiding in South Sudan by:
- Simulating herder navigation challenges
- Modeling theft detection scenarios  
- Testing automated alert systems
- Evaluating AI-driven monitoring solutions

The RL agent learns optimal strategies for:
- Efficient cattle registration
- Theft detection and reporting
- Avoiding dangerous encounters
- Maximizing mission success rates

## Author

**Geu Aguto Garang Bior**  
BSc (Hons) Software Engineering  
African Leadership University  
ML Techniques II - Summative Assignment

## License

This project is developed for academic purposes as part of the ML Techniques II course at African Leadership University.