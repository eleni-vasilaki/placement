# Q-Learning in Grid World: Starter Notebook for Timucin

import numpy as np
import matplotlib.pyplot as plt
import random

# Define the grid world
# 0: empty cell
# -1: wall
# +1: goal

grid = np.array([
    [ 0,  0,  0,  1],
    [ 0, -1,  0, -1],
    [ 0,  0,  0,  0]
])

n_rows, n_cols = grid.shape

actions = ['up', 'down', 'left', 'right']

def is_terminal(state):
    r, c = state
    return grid[r, c] == 1

def get_start():
    return (2, 0)  # bottom-left corner

def get_next_state(state, action):
    r, c = state
    if action == 'up': r = max(r - 1, 0)
    elif action == 'down': r = min(r + 1, n_rows - 1)
    elif action == 'left': c = max(c - 1, 0)
    elif action == 'right': c = min(c + 1, n_cols - 1)
    if grid[r, c] == -1:
        return state  # hit wall, stay in place
    return (r, c)

def get_reward(state):
    r, c = state
    return grid[r, c]

# Initialise Q-table with small random values to avoid misleading tie-breaks
Q = {}
for r in range(n_rows):
    for c in range(n_cols):
        if grid[r, c] != -1:
            Q[(r, c)] = {a: np.random.uniform(0, 0.01) for a in actions}

# Hyperparameters
epsilon = 0.2  # Exploration rate
alpha = 0.5    # Learning rate
gamma = 0.9    # Discount factor

# Placeholder for storing rewards
steps_log = []

# Q-learning algorithm

def train_q_learning(episodes=500):
    for episode in range(episodes):
        state = get_start()
        total_reward = 0
        steps = 0
        while not is_terminal(state):
            # Choose action
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)

            next_state = get_next_state(state, action)
            reward = get_reward(next_state)
            total_reward += reward

            max_next = max(Q[next_state].values())

            # TODO: Implement Q-value update here

            state = next_state
            steps += 1
        steps_log.append(steps)


# Visualisation
def print_policy():
    direction_map = {'up': '^', 'down': 'v', 'left': '<', 'right': '>'}
    for r in range(n_rows):
        row = ''
        for c in range(n_cols):
            if grid[r, c] == -1:
                row += '### '
            elif grid[r, c] == 1:
                row += ' G  '
            else:
                q_vals = Q[(r, c)]
                best_action = max(q_vals, key=q_vals.get)
                values = list(q_vals.values())
                if all(abs(val - values[0]) < 1e-6 for val in values):
                    row += ' ?  '  # Indeterminate: no learning occurred
                else:
                    row += f' {direction_map[best_action]}  '
        print(row)

# Learning curve plot
# Learning curve plot (steps to reach goal)
def plot_learning_curve():
    plt.plot(steps_log)
    plt.xlabel('Episode')
    plt.ylabel('Steps to Reach Goal')
    plt.title('Learning Curve (SARSA)')
    plt.grid(True)
    plt.show()

# Train and evaluate
train_q_learning()
print("Learned policy:")
print_policy()
plot_learning_curve()

# Suggestions:
# - Modify the grid layout to make the world larger or more complex
# - Tune epsilon, alpha, and gamma to see how learning changes
# - Try adding negative rewards to some cells to simulate traps or penalties
# - After filling in the Q-update line, observe how the agent learns over time
