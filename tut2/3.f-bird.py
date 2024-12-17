#https://www.youtube.com/watch?v=RVMpm86equc

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):    
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)       
    
    def act(self, state, epsilon):    
        if random.random() > epsilon:
            with torch.no_grad():
                return self.forward(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

render = True
is_training = True

from replay_memory import ReplayMemory
import itertools
import gymnasium as gym

# hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_DECAY = 0.9995
EPS_MIN = 0.05
ReplayMemory_SIZE = 10000


if __name__ == '__main__':
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions).to(device)
    #target_net = DQN(n_observations, n_actions).to(device)
    #target_net.load_state_dict(policy_net.state_dict())

    if is_training:
        memory = ReplayMemory(ReplayMemory_SIZE)
        epsilone = EPS_START


    episode_durations = []
    rewards_per_episode = []
    epsilone_history = []
    for episode in itertools.count(): # infinite loop
        t = 0
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)

        terminated = truncated = False
        episode_reward = 0.0

        while not terminated and not truncated:
            t += 1

            if is_training and random.random() < epsilone:
                action = env.action_space.sample()
                action = torch.tensor([action], device=device, dtype=torch.long)
            else:
                with torch.no_grad():
                    action = policy_net(state.unsqueeze(dim=0)).squeeze().argmax()
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            new_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            reward = torch.tensor([reward], device=device)

            memory.push(torch.tensor([state], device=device),
                        torch.tensor([action], device=device),
                        torch.tensor([next_state], device=device),
                        torch.tensor([reward], device=device),
                        torch.tensor([terminated], device=device))

            episode_reward += reward

            state = next_state


            if is_training:
                memory.append((state, action, next_state, reward, terminated))
                
            episode_durations.append(t + 1)

        rewards_per_episode.append(episode_reward)
        epsilone = max(EPS_MIN, epsilone * EPS_DECAY)
        epsilone_history.append(epsilone)

    env.close()

    print('done')