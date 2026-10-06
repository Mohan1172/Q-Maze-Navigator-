# Maze Navigation Agent using Q-Learning

## Project Overview

This project implements a Maze Navigation Agent using Reinforcement Learning. The agent learns to navigate through a 6x6 maze from a starting position to a goal while avoiding obstacles.

The project uses Q-Learning to learn the best action for each state. After training, the learned agent is visualized using Pygame.

## Features

- 6x6 maze environment
- Q-Learning based agent
- Obstacle avoidance
- Reward-based learning
- Epsilon-greedy action selection
- Q-table based learning
- 600 training episodes
- Pygame visualization
- Automatic maze navigation

## Technologies Used

- Python
- NumPy
- Pygame
- Reinforcement Learning
- Q-Learning

## Project Structure

```text
Maze-RL/
│
├── environment.py
├── main.py
└── README.md
```

## Maze Environment

The project uses a 6x6 grid maze.

- Start Position: `(0, 0)`
- Goal Position: `(5, 5)`
- Actions: Up, Down, Left, Right
- Obstacles are predefined inside the maze.

## Reward System

| Situation | Reward |
|-----------|--------|
| Reach the goal | +100 |
| Hit an obstacle | -10 |
| Normal movement | -1 |

## Q-Learning Agent

The agent learns the best action for each state using a Q-table.

### Parameters

```text
Learning Rate = 0.1
Discount Factor = 0.95
Initial Epsilon = 1.0
Epsilon Decay = 0.995
Minimum Epsilon = 0.01
```

The agent uses an epsilon-greedy strategy to balance exploration and exploitation.

## Training

The agent is trained for 600 episodes.

During each episode:

1. The maze environment is reset.
2. The current state is obtained.
3. The agent selects an action.
4. The environment returns the next state and reward.
5. The Q-table is updated.
6. Epsilon is gradually reduced.
7. The process continues until the goal is reached or the step limit is reached.

A maximum of 100 steps is allowed in each episode.

## Pygame Visualization

After training, the learned Q-table is used to control the agent.

The Pygame visualization displays:

- Blue = Agent
- Red = Obstacles
- Green = Goal
- Grid = Maze

The agent automatically follows the learned policy to reach the goal.

## Installation

Install the required libraries:

```bash
pip install numpy pygame
```

## How to Run

Open the project folder in VS Code and run:

```bash
python main.py
```

The program will:

1. Train the Q-Learning agent for 600 episodes.
2. Display training progress in the terminal.
3. Print the learned Q-table.
4. Open the Pygame simulation.
5. Automatically navigate the maze.

## Key Concepts

- Reinforcement Learning
- Q-Learning
- State Space
- Action Space
- Reward Function
- Q-Table
- Epsilon-Greedy Strategy
- Exploration and Exploitation
- Autonomous Navigation
- Environment Simulation

## Learning Outcome

This project demonstrates how an agent can learn to navigate an environment using rewards and penalties instead of being given a fixed path.

The agent gradually learns which actions are useful for reaching the goal while avoiding obstacles.

## Future Improvements

- Random maze generation
- Larger maze environments
- Multiple goals
- Moving obstacles
- Training performance graphs
- Reward and step tracking
- Comparison with other Reinforcement Learning algorithms
- Graphical training-progress dashboard

## Author

**Mohan Dara Krishna Juvvala**

B.Tech - Computer Science Engineering (AI & ML)
