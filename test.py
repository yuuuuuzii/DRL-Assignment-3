import torch
import numpy as np
import gym
import os
from DQN_agent import DQNAgent
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from collections import deque
from student_agent import Agent
import argparse

# === 初始化 agent ===
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
action_size = env.action_space.n
state_size = env.observation_space.shape

agent = Agent()

# === 測試 function ===
num_episodes = 10
all_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        # env.render()
        state = next_state
        steps += 1  

    all_rewards.append(total_reward)
    print(f"Episode {episode + 1} finished with reward {total_reward}")
print(sum(all_rewards)/10)
env.close()
