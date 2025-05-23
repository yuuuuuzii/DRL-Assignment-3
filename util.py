import torch
import random
from collections import deque
import ipdb
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
import wandb
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
class QNet(nn.Module):
    def __init__(self, n_actions, hidden_dim=512, sigma_init=0.5):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.conv_out_size = 7*7*128

        # Value stream
        self.value_fc1 = NoisyLinear(self.conv_out_size, hidden_dim, sigma_init)
        self.value_fc2 = NoisyLinear(hidden_dim, 1, sigma_init)

        # Advantage stream
        self.advantage_fc1 = NoisyLinear(self.conv_out_size, hidden_dim, sigma_init)
        self.advantage_fc2 = NoisyLinear(hidden_dim, n_actions, sigma_init)
        
    def forward(self, x):
        x = self.feature_extractor(x)

        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        q = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

class ICM(nn.Module):
    def __init__(self, n_actions, conv_out_size, encoder: nn.Module):
        super().__init__()
        # 直接接收 QNet 裡的 feature_extractor
        self.encoder = encoder

        # inverse model: 輸入是 concat(hidden_s, hidden_s')
        self.inverse_fc = nn.Sequential(
            nn.Linear(conv_out_size*2, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        # forward model: 輸入是 concat(hidden_s, onehot(a))
        self.forward_fc = nn.Sequential(
            nn.Linear(conv_out_size + n_actions, 512),
            nn.ReLU(),
            nn.Linear(512, conv_out_size)
        )

    def forward(self, state, next_state, action):
        # state, next_state: [B, C, H, W], action: [B] int64
        hidden_state  = self.encoder(state)        # [B, feature_dim]
        hidden_next_state = self.encoder(next_state) # [B, feature_dim]
            
        # inverse loss
        inv_input = torch.cat([hidden_state, hidden_next_state], dim=1)               # [B, 2*feature_dim]
        logits_a  = self.inverse_fc(inv_input)                  # [B, n_actions]
        inverse_loss = F.cross_entropy(logits_a, action)

        # forward loss
        a_onehot = F.one_hot(action, logits_a.size(1)).float()  # [B, n_actions]
        fwd_input = torch.cat([hidden_state, a_onehot], dim=1)           # [B, feature_dim+n_actions]
        pred_hidden_next_state = self.forward_fc(fwd_input)                  # [B, feature_dim]

        diff = pred_hidden_next_state - hidden_next_state
        forward_loss = diff.abs().mean(dim=1).mean()
        intrinsic_reward = diff.abs().sum(dim=1)          # [B,]

        return inverse_loss, forward_loss, intrinsic_reward
    

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=1e-4, epsilon=1e-6, device=None):
        self.capacity = capacity
        self.buffer = [] # 這邊不用deque是因為要根據 priority 來 sample
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0 ##紀錄滿了之後要存到哪

        self.alpha = alpha                        # priority exponent
        self.beta = beta                          # importance sampling exponent
        self.beta_increment = beta_increment_per_sampling # beta 要慢慢變大

        self.epsilon = epsilon                    # small amount to avoid zero priority
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        # Convert numpy arrays to tensors
        state = torch.tensor(np.array(state), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)

        transition = (state, action, reward, next_state, done)

        ##先找最大的priority, 然後讓最新進的資料有最大的priority, 至少先跑一次
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        # Compute normalized priorities
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.alpha
        probs /= probs.sum()

        # Sample indices according to probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*samples)
        states      = torch.stack(states).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)

        #先產生一個 wi 的weight陣列 
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        # td_errors: tensor or array of shape [batch_size]
        # 不用1/rank是因為計算量太大＝＝
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(float(td)) + self.epsilon

    def __len__(self):
        return len(self.buffer)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.2):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # μ parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        # σ parameters
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        # ε buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        # 初始化 μ
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # 初始化 σ
        self.weight_sigma.data.fill_(self.sigma_init/math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init/math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        # factorized gaussian noise
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        # outer product for weights, direct for bias
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
            # sigma_weight = torch.clamp(, min=0.02)
            # sigma_bias   = torch.clamp(self.bias_sigma, min=0.02)
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon

        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
      
    
class LittleBuffer:
    def __init__(self, n_step = 5, gamma = 0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque()

    def add(self, transition):
        self.buffer.append(transition)

    def is_full(self):
        return len(self.buffer) == self.n_step
    
    def get_n_step_transition(self):

            total_reward = 0.0
            next_state = None

            # 計算累積 reward，並找出第一個 done 的位置（若有）
            for idx, (s, a, r, ns, done) in enumerate(self.buffer):
                total_reward += (self.gamma ** idx) * r
                if done:
                    next_state = ns
                    done_flag = True
                    break

            if next_state is None:
                state, action, _, _, _ = self.buffer[0]
                _, _, _, next_state, done_flag = self.buffer[self.n_step - 1]
            else:
                # 有遇到 done 時，state/action 還是 buffer[0]
                state, action, _, _, _ = self.buffer[0]

            return (state, action, total_reward, next_state, done_flag)


    def pop(self):
        self.buffer.popleft()

    def clear(self):
        self.buffer.clear()


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)
    env = NormalizeObservation(env) 
    return env

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.obs_buffer = deque(maxlen=2)  # 用來存最近2個obs

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # 做 pixel-wise max
        max_frame = np.maximum(self.obs_buffer[0], self.obs_buffer[-1])
        return max_frame, total_reward, done, info

    def reset(self):
        self.obs_buffer.clear()
        obs = self.env.reset()
        self.obs_buffer.append(obs)
        return obs

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0