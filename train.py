import numpy as np
from DQN_agent import DQNAgent
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import ipdb
import wandb
from tqdm import tqdm
from collections import deque
from util import make_env


def main():
    env = make_env()

    action_size = env.action_space.n
    agent = DQNAgent(action_size)
    num_episodes = agent.episode
    reward_history = []

    wandb.init(
        project="mario-dqn",         
        name="run_with_icm_per",    
        config={                  
            # "lr": agent.lr,
            # "tau": agent.tau,
            "sigma_init": agent.sigma_init
        }
    )
    print(f"lr = {agent.lr:5f}, tau = {agent.tau:5f}, sigma_init = {agent.sigma_init}")
    for episode in tqdm(range(num_episodes)):
        state = env.reset() 
        agent.episode = episode
        done = False
        total_reward = 0
        second_stage = 0
        # 每回合的步驟
        while not done:
    
            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)
            if info.get('flag_get', False):
                second_stage += 1
            

            agent.little_buffer.add((state.copy(), action, reward, next_state.copy(), done))

            if agent.little_buffer.is_full():
                n_step_transition = agent.little_buffer.get_n_step_transition()
                agent.buffer.add(*n_step_transition)  # 拆開來塞進去
                agent.little_buffer.pop() 

            if done:
                while len(agent.little_buffer.buffer) > 0:
                    n_step_transition = agent.little_buffer.get_n_step_transition()
                    agent.buffer.add(*n_step_transition)
                    agent.little_buffer.pop() 
                agent.little_buffer.clear()
    
            if len(agent.buffer.buffer) > agent.batchsize:
                agent.train()
                agent.step_count += 1
                if agent.step_count % agent.update_frequency == 0:
                    agent.update()

            state = next_state  
            total_reward += reward
            #env.render()

        avg_q   = np.mean(agent.q_loss_log)
        avg_inv = np.mean(agent.inverse_loss_log)
        avg_fwd = np.mean(agent.forward_loss_log)

        # # 計算 raw intrinsic/extrinsic mean 及 scaled intrinsic
        intr_mean_raw    = np.mean(agent.intrinsic_reward)
        extr_mean        = np.mean(agent.rewards_e)
        intr_mean_scaled = agent.eta * intr_mean_raw

        total = avg_q + agent.alpha * avg_inv + agent.beta * avg_fwd

        #紀錄
        wandb.log({
            "loss/q_loss":           avg_q,
            "loss/inverse":          agent.alpha * avg_inv,
            "loss/forward":          agent.beta  * avg_fwd,
             "loss/total":            total,
            "reward/episode":        total_reward,
            "reward/intrinsic_raw":  intr_mean_raw,
            "reward/intrinsic_scaled": intr_mean_scaled,
            "reward/extrinsic_mean": extr_mean,
            "advantage_fc1_sigma": agent.q_net.advantage_fc1.weight_sigma.mean().item(),
            "advantage_fc2_sigma": agent.q_net.advantage_fc2.weight_sigma.mean().item(),
            "value_fc1_sigma": agent.q_net.value_fc1.weight_sigma.mean().item(),
            "value_fc2_sigma": agent.q_net.value_fc2.weight_sigma.mean().item(),
            "step":                  episode,
        })

        # 清空每集累積的 list
        agent.q_loss_log.clear()
        agent.inverse_loss_log.clear()
        agent.forward_loss_log.clear()
        agent.intrinsic_reward.clear()
        agent.rewards_e.clear()
            # 更新狀態和總回報
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Second_stage: {second_stage}") #, epsilon: {agent.epsilon:.4f}
        
        reward_history.append(total_reward)

        if (episode+1) % 100 == 0:
            checkpoint_prefix = f"/home/bl530/Desktop/DRL/DRL-Assignment-3/mario_episode_{episode + 1}"
            agent.save_checkpoint(checkpoint_prefix)

        # 嘗試從 checkpoint 載入



if __name__ == "__main__":
    main()