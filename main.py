import numpy as np
import random
import pygame
import sys

# ==========================================
# 1. ENVIRONMENT DEFINITION
# ==========================================
class MazeEnv:
    def __init__(self, size=6):
        self.size = size
        # 0: Empty, 1: Obstacle/Wall, 2: Goal
        self.grid = np.zeros((size, size), dtype=int)
        
        # Obstacles (Row, Col)
        self.obstacles = [
            (1, 1), (1, 2), (2, 4), 
            (3, 1), (3, 2), (4, 4), (4, 1)
        ]
        for r, c in self.obstacles:
            self.grid[r, c] = 1
            
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.grid[self.goal_pos] = 2
        
        # 4 Discrete Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = 4
        self.state_space = size * size
        self.agent_pos = self.start_pos

    def pos_to_state(self, pos):
        """Convert 2D coordinate to 1D state index."""
        return pos[0] * self.size + pos[1]

    def reset(self):
        """Reset the agent back to the starting point."""
        self.agent_pos = self.start_pos
        return self.pos_to_state(self.agent_pos)

    def step(self, action):
        """Transition dynamic: state + action -> next_state, reward, done."""
        r, c = self.agent_pos
        
        if action == 0:    # Up
            r = max(0, r - 1)
        elif action == 1:  # Down
            r = min(self.size - 1, r + 1)
        elif action == 2:  # Left
            c = max(0, c - 1)
        elif action == 3:  # Right
            c = min(self.size - 1, c + 1)

        next_pos = (r, c)
        
        # Reward function design
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


# ==========================================
# 2. Q-LEARNING AGENT
# ==========================================
class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha                # Learning rate
        self.gamma = gamma                # Discount factor
        self.epsilon = epsilon            # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table (States x Actions)
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        """Epsilon-greedy decision policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)  # Explore
        return np.argmax(self.q_table[state])                 # Exploit

    def update(self, state, action, reward, next_state, done):
        """Bellman Optimality update equation."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state, best_next_action])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_agent(episodes=600):
    env = MazeEnv(size=6)
    agent = QLearningAgent(env.state_space, env.action_space)

    print(f"--- Starting Training ({episodes} episodes) ---")
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        if ep % 100 == 0:
            print(f"Episode {ep:3d}/{episodes} | Total Reward: {total_reward:6.1f} | Epsilon: {agent.epsilon:.3f}")

    print("--- Training Complete! ---")
    return env, agent


# ==========================================
# 4. INTERACTIVE PYGAME SIMULATION
# ==========================================
def run_interactive_simulation(env, agent):
    pygame.init()
    cell_size = 80
    width = env.size * cell_size
    height = env.size * cell_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Maze Navigation Agent")
    clock = pygame.time.Clock()

    WHITE  = (245, 245, 245)
    BLUE   = (41, 128, 185)   # Agent
    RED    = (192, 57, 43)    # Obstacles
    GREEN  = (39, 174, 96)    # Goal
    GRAY   = (210, 210, 210)  # Gridlines

    state = env.reset()
    done = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Deterministic exploitation of the learned Q-table
        if not done:
            action = np.argmax(agent.q_table[state])
            state, reward, done = env.step(action)

        # Render maze layout
        screen.fill(WHITE)
        for r in range(env.size):
            for c in range(env.size):
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, GRAY, rect, 1)

                if (r, c) in env.obstacles:
                    pygame.draw.rect(screen, RED, rect)
                elif (r, c) == env.goal_pos:
                    pygame.draw.rect(screen, GREEN, rect)

        # Render Agent position
        ar, ac = env.agent_pos
        agent_center = (ac * cell_size + cell_size // 2, ar * cell_size + cell_size // 2)
        pygame.draw.circle(screen, BLUE, agent_center, cell_size // 3)

        pygame.display.flip()
        clock.tick(3)  # 3 frames per second

        if done:
            pygame.time.wait(2000)
            state = env.reset()
            done = False


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    env, agent = train_agent(episodes=600)
    
    # Print the full learned Q-Table
    print("\nLearned Q-Table (States x Actions [0:Up, 1:Down, 2:Left, 3:Right]):")
    print(np.round(agent.q_table, 2))
    print("-" * 60)
    
    # Launch Pygame simulation
    run_interactive_simulation(env, agent)