import numpy as np

class MazeEnv:
    def __init__(self, size=6):
        self.size = size
        # 0: Empty, 1: Obstacle/Wall, 2: Goal
        self.grid = np.zeros((size, size), dtype=int)
        
        # Define obstacle positions (Row, Col)
        self.obstacles = [
            (1, 1), (1, 2), (2, 4), 
            (3, 1), (3, 2), (4, 4), (4, 1)
        ]
        for r, c in self.obstacles:
            self.grid[r, c] = 1
            
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.grid[self.goal_pos] = 2
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = 4
        self.state_space = size * size
        self.agent_pos = self.start_pos

    def pos_to_state(self, pos):
        """Convert 2D (row, col) coordinates to a single discrete state index."""
        return pos[0] * self.size + pos[1]

    def reset(self):
        """Reset the agent back to the starting point."""
        self.agent_pos = self.start_pos
        return self.pos_to_state(self.agent_pos)

    def step(self, action):
        """Executes one step in the environment."""
        r, c = self.agent_pos
        
        # Movement mapping
        if action == 0:    # Up
            r = max(0, r - 1)
        elif action == 1:  # Down
            r = min(self.size - 1, r + 1)
        elif action == 2:  # Left
            c = max(0, c - 1)
        elif action == 3:  # Right
            c = min(self.size - 1, c + 1)

        next_pos = (r, c)
        
        # Evaluate reward and termination
        if next_pos in self.obstacles:
            reward = -10.0
            done = False
        elif next_pos == self.goal_pos:
            reward = 100.0
            done = True
            self.agent_pos = next_pos
        else:
            reward = -1.0
            done = False
            self.agent_pos = next_pos

        return self.pos_to_state(self.agent_pos), reward, done