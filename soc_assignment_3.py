import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Create the Taxi environment
env = gym.make('Taxi-v3')

# Hyperparameters
num_episodes = 3000
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon for epsilon-greedy policy
alpha = 0.1  # Learning rate

# File paths to save/load the Q-table
q_table_npy_file = "q_table_ql.npy"
q_table_json_file = "q_table_ql.json"

# Initialize Q-table for Taxi environment
q_table_ql = np.zeros((env.observation_space.n, env.action_space.n))

# Function to choose action based on epsilon-greedy policy
def choose_action(state, q_table, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

# Q-Learning algorithm
def q_learning(env, q_table, num_episodes, gamma, alpha, epsilon):
    cumulative_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()  # Unpack initial state
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action]
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error
            state = next_state
            total_reward += reward
            
            done = terminated or truncated
        
        cumulative_rewards.append(total_reward)

    return cumulative_rewards

# Function to save the Q-table in npy format
def save_q_table_npy(file_path, q_table):
    np.save(file_path, q_table)

# Function to load the Q-table from npy format
def load_q_table_npy(file_path):
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        return np.zeros((env.observation_space.n, env.action_space.n))

# Function to save the Q-table in json format
def save_q_table_json(file_path, q_table):
    with open(file_path, 'w') as f:
        json.dump(q_table.tolist(), f)

# Function to load the Q-table from json format
def load_q_table_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return np.array(json.load(f))
    else:
        return np.zeros((env.observation_space.n, env.action_space.n))

# Load existing Q-table if available
q_table_ql = load_q_table_npy(q_table_npy_file)

# Train Q-Learning if no pre-trained Q-table is found
cumulative_rewards_ql = []
if not os.path.exists(q_table_npy_file):
    cumulative_rewards_ql = q_learning(env, q_table_ql, num_episodes, gamma, alpha, epsilon)
    save_q_table_npy(q_table_npy_file, q_table_ql)
    save_q_table_json(q_table_json_file, q_table_ql)
else:
    cumulative_rewards_ql = q_learning(env, q_table_ql, num_episodes, gamma, alpha, epsilon)
# Save the Q-table in both formats
save_q_table_npy(q_table_npy_file, q_table_ql)
save_q_table_json(q_table_json_file, q_table_ql)

# Plot cumulative rewards if available
if cumulative_rewards_ql:
    plt.plot(range(num_episodes), cumulative_rewards_ql, label="Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Rewards for Taxi Environment (Q-Learning)")
    plt.legend()
    plt.show()
