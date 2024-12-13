import gymnasium as gym
from collections import defaultdict # allow access to keys do not exist
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # draw shapes
import numpy as np
import seaborn as sns
from tqdm import tqdm

env = gym.make('Blackjack-v1', 
               sab=True, 
               render_mode="rgb_array")

# reset the env
done=False
observation, info = env.reset()

# objservation = (16,9,False)

player_sum, dealer_faceup, player_usable_ace = observation
print('==============================')
print(f'player_sum={player_sum}')
print(f'player_usable_ace={player_usable_ace}')
print(f'dealer_faceup={dealer_faceup}')

for i in range(1000):
    action = env.action_space.sample()
    print('==============================')
    print(f'action[{i}]={action}')
    observation,reward,terminated,truncated,info =  env.step(action)

    player_sum, dealer_faceup, player_usable_ace = observation
    print('==============================')
    print(f'player_sum={player_sum}')
    print(f'player_usable_ace={player_usable_ace}')
    print(f'dealer_faceup={dealer_faceup}')

    print(f'reward={reward}')
    print(f'terminated={terminated}')
    print(f'truncated={truncated}')

    if terminated == True:
      print('Terminated')
      break

    if truncated == True:
      print('Truncated!')
      break

print('done')