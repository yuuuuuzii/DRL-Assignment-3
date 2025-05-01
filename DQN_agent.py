import torch
from util import QNet, PrioritizedReplayBuffer , LittleBuffer , NoisyLinear
import numpy as np
import ipdb
import os
import wandb
import math

class DQNAgent: # 用Double DQN
    def __init__(self,action_size):
        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.
      
        self.action_size = action_size
        self.batchsize = 256
        self.buffer_size = 50000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_loss_log = []
        self.inverse_loss_log = []
        self.forward_loss_log = []
        self.intrinsic_reward = []
        self.rewards_e = []
        self.step_count = 0
        self.sigma_init = 2.5
        self.episode = 5000
        self.lr = 0.0008
        self.update_frequency = 1
        self.tau = 0.000005
        self.alpha = 0.15
        self.beta = 0.7
        self.gamma = 0.99
        self.q_net = QNet(self.action_size, 512 ,self.sigma_init).to(self.device)
        self.target_net = QNet(self.action_size, 512 ,self.sigma_init).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.buffer = PrioritizedReplayBuffer(self.buffer_size)
        self.little_buffer = LittleBuffer(n_step=9, gamma= self.gamma)
        self.optimizer= torch.optim.Adam(list(self.q_net.parameters()), lr=self.lr)

        # 
        # self.eta = 0.1
        # self.clip = 5
        # 共用 q_net 的 feature_extractor
        # self.icm = ICM(n_actions=self.action_size, feature_dim = 512, encoder=self.q_net.feature_extractor).to(self.device)

        
        # encoder_params = set(self.q_net.feature_extractor.parameters())
        # self.icm_only_params = [p for p in self.icm.parameters() if p not in encoder_params]

        
        # self.optimizer2 = torch.optim.Adam(self.icm_only_params, lr=self.lr)

    def get_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device) #把state 轉為 [1, dimension]的維度
        with torch.no_grad():
            action_values = self.q_net(state)  # 計算 Q_net預測的 action 分佈
        return torch.argmax(action_values, dim=1).item()

    def update(self):
        for p_target, p_online in zip(self.target_net.parameters(), self.q_net.parameters()):
            p_target.data.copy_(self.tau * p_online.data + (1 - self.tau) * p_target.data)
        #self.target_net.load_state_dict(self.q_net.state_dict())

    # def compute_sigma_loss(self, net):
    #     sigma_loss = 0.0
    #     for m in net.modules():
    #         if isinstance(m, NoisyLinear):
    #             # 懲罰接近 0 的 sigma
    #             penalty = (0.5 - m.weight_sigma).clamp(min=0)
    #             sigma_loss += penalty.mean()
    #     return self.sigma_penalty * sigma_loss
    
    def train(self):
        # TODO: Sample a batch from the replay buffer
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batchsize)

        # inverse_loss, forward_loss, intrinsic_reward = self.icm(states, next_states, actions)
        
        # self.intrinsic_reward.append(intrinsic_reward.mean().item())
        self.rewards_e.append(rewards.mean().item())
        # intrinsic reward weight
        # rewards = rewards_e + self.eta * intrinsic_reward
        
        # TODO: Compute loss and update the model
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 用 online network (q_net) 選出最佳行動
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)

            max_next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        
        td_errors = (target_q_values - q_values)
        td_errors_for_priority = td_errors.detach()

        #loss = (weights * td_errors.pow(2)).mean()

        huber = torch.nn.functional.smooth_l1_loss(
            td_errors,
            torch.zeros_like(td_errors),
            reduction='none',
            beta=1.0
        )
        loss = (weights * huber).mean()
        
        # loss = q_loss + self.alpha * inverse_loss + self.beta * forward_loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm = self.clip)
        self.optimizer.step()
        self.buffer.update_priorities(indices, td_errors_for_priority)

        self.q_loss_log.append(loss.item())
        # self.inverse_loss_log.append(inverse_loss.item())
        # self.forward_loss_log.append(forward_loss.item())

    def reset_noisy_sigma(self):
            for net in (self.q_net, self.target_net):
                for module in net.modules():
                    if isinstance(module, NoisyLinear):
                        # 只 reset sigma，不動 μ
                        module.weight_sigma.data.fill_(
                            module.sigma_init / math.sqrt(module.in_features)
                        )
                        if module.bias_sigma is not None:
                            module.bias_sigma.data.fill_(
                                module.sigma_init / math.sqrt(module.out_features)
                            )
                            
    def save_checkpoint(self, path_prefix):
        dirname = os.path.dirname(path_prefix)
        os.makedirs(dirname, exist_ok=True)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            #'ICM': self.icm.state_dict(),
            'step_count': self.step_count
        }, path_prefix + '_checkpoint.pth')
        print(f"✅ Checkpoint saved: {path_prefix}_checkpoint.pth")

    def load_checkpoint(self, path_prefix):
        path = path_prefix + '_checkpoint.pth'
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ No checkpoint found at: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']
        #self.icm = checkpoint['ICM']
        print(f"✅ Checkpoint loaded from: {path}")

    