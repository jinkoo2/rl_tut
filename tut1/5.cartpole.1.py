import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

device = torch.device('cpu')

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))

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

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer1 = nn.Linear(128, 128)
        self.layer1 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

batch_size = 128 # number of transitions to be batched.
gamma = 0.99 # discount factor
esp_start = 0.9 # final epsilon
eps_end = 0.05 # start epsilon
eps_decay = 1000 # rate for expenential decay, higher the slower the decay becomes
tau = 0.005 # update rate of target network
lr = 1e-4 #3 learning rate (AdamW optimizer)

n_actions = env.action_space.n
print(f'n_actions={n_actions}')

state, info = env.reset()

# number of features (variables) in the state
n_observation = len(state)
print(f'n_observation={n_observation}')

# policy and target sets with same parameters
policy_net = DQN(n_observations=n_observation, n_actions=n_actions).to(device)
target_net = DQN(n_observations=n_observation, n_actions=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=lr)

memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (esp_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
    steps_done +=1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]]).to(device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    duration_t = torch.tensor(episode_durations, dtype=torch.long)
    if show_result:
        plt.title('Result')
    else:
        plt.clf() 
        plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(duration_t.numpy)

    # moving average
    if len(duration_t)>100:
        means = duration_t.unfold(0, 1000, 1).mean(1).view(1,1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001) # pause so that plots are updated

def optimize_model():
    if len(memory) < batch_size:
        return 
    
    transition = memory.sample(batch_size)
    
    # convert batch-array of Transitions to Transitions of batch array
    batch = Transition(*zip(*transition))

    


print('done')





