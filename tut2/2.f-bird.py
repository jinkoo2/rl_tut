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


   
if __name__ == '__main__':
    n_objervations = 12
    n_actions = 2
    net = DQN(n_objervations, n_actions)
    state = torch.rand((10, n_objervations))
    print(f'state={state.shape}')
    output = net(state)
    print(f'output={output.shape}')

    print('done')