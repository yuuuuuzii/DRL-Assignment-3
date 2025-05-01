import torch
import gym
import numpy as np
from DQN_agent import DQNAgent
import cv2
from collections import deque
    
class Agent(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = gym.spaces.Discrete(12)
        self.agent = DQNAgent(action_size=12)
        self.agent.load_checkpoint('mario_episode_500')
        self.frame_stack = deque(maxlen=4)
        self.action_buffer = []

    def _preprocess(self, observation):
        """
        Convert raw RGB observation to (4,84,84) tensor:
         - Grayscale
         - Resize to 84×84
         - Append to internal deque and stack last 4 frames
        """

        arr = np.asarray(observation)
        resized = cv2.resize(arr, (84, 84), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        self.frame_stack.append(gray)

        while len(self.frame_stack) < self.frame_stack.maxlen:
            self.frame_stack.append(gray)

        # Stack
        stacked = np.stack(self.frame_stack, axis=0)
        # To tensor
        tensor = torch.tensor(stacked, dtype=torch.float32)/255.0
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def act(self, observation):
        # Preprocess raw obs into (4,84,84)
        if len(self.action_buffer) != 0 :
            return self.action_buffer.pop()
        
        state = self._preprocess(observation)
      
        self.agent.q_net.reset_noise()

        with torch.no_grad():
     
            q = self.agent.q_net(state)
            for _ in range(4):
                self.action_buffer.append(q.argmax(dim=1).item())
        # Greedy action

        return self.action_buffer.pop()
