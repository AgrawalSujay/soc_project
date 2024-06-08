import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Reward distributions for the arms


def arm_1():
    return np.random.normal(0, 1)

def arm_2():
    if np.random.rand() < 0.5:
        return 3  
    else:
        return -4

def arm_3():
    return poisson.rvs(2)

def arm_4():
    return np.random.normal(1, np.sqrt(2))

def arm_5():
    return np.random.exponential(1)

def arm_6():
    return arms[np.random.randint(0,4)]()

arms = [arm_1,arm_2, arm_3, arm_4, arm_5, arm_6]

# ε-Greedy Algorithm
def epsilon_greedy(epsilon):
    num_arms = 6
    rewards_per_episode = []

    for i in range(1000):
        Q = np.zeros(num_arms)
        N = np.zeros(num_arms)
        total_reward = 0

        for j in range(100):
            if np.random.rand() < epsilon:
                action = np.random.randint(0,5)
            else:
                max=int(-1)
                for k in range(6):
                    if Q[k]>max:
                        action = k
            reward = arms[action]()
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
    
    return rewards_per_episode


epsilons = [0.1, 0.01, 0]

results = {}
for epsilon in epsilons:
    rewards = epsilon_greedy(epsilon)
    results[epsilon] = rewards

plt.figure(figsize=(12, 8))
for epsilon,rewards in results.items():
    plt.plot(rewards)

plt.xlabel('Episode')
plt.ylabel('Reward at the end of episode')
plt.title('Reward at the end of episode vs Episode for different ε-Greedy strategies')
plt.legend()
plt.show()
