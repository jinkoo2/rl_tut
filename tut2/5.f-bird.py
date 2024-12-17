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


device = 'gpu' if torch.cuda.is_available() else 'cpu'
import torch.optim as optim
from collections import namedtuple, deque

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))
from itertools import count

import gymnasium as gym

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        ''' Save a transition '''
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
if __name__ == '__main__':
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
    memory = ReplayMemory(10000)

    episode_durations = []

    for episode in range(1000):
        state, info = env.reset()
        for t in count():
            action = policy_net.act(torch.tensor([state], device=device), 0.05)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            memory.push(torch.tensor([state], device=device),
                        torch.tensor([action], device=device),
                        torch.tensor([next_state], device=device),
                        torch.tensor([reward], device=device),
                        torch.tensor([terminated], device=device))
            state = next_state
            if terminated or truncated:
                episode_durations.append(t + 1)
                break

    print(f'episode_durations={episode_durations}')
    env.close()
