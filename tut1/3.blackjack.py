import gymnasium as gym
from collections import defaultdict # allow access to keys do not exist
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # draw shapes
import numpy as np
import seaborn as sns
from tqdm import tqdm

env = gym.make('Blackjack-v1', sab=True, render_mode="rgb_array")


class BlackjackAgent:
    def __init__(self, learning_rate: float, 
        initial_epsilon: float, 
        epsilon_decay:float, 
        final_epsilon:float, 
        discount_factor: float = 0.95):
      
      print('Initialzing with empty dictionary of state-action values (q_values), a learning rate, and an epsilon.')
      n = env.action_space.n # n=2
      zeros = lambda:np.zeros(n) # lamda function that returns a default value, which is [0,0]
      self.q_values = defaultdict(zeros) # when key is not present, the deault dict will return a numpy array of [0,0].

      print(f'n={n}')
      print(f'zeros={zeros}')
      print(f'self.q_values={self.q_values}')

      self.learning_rate = learning_rate
      self.discount_factor = discount_factor
      self.epsilon = initial_epsilon
      self.epsilon_decay = epsilon_decay
      self.final_epsilon = final_epsilon

      self.training_error = []

    def get_action(self, obs: tuple[int, int, bool])->int:
      r = np.random.random()
      if r < self.epsilon:
         #print('exploring!')
         return env.action_space.sample()
      else:
         #print('exploiting!')
         a = self.q_values[obs]
         #a[action=0] = exected reward for action 0
         #a[action=1] = exected reward for action 1
        
         action_of_max_reward = np.argmax(a) # return the action of the maximum probability (reward)

         return int(action_of_max_reward)

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.learning_rate * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
       self.epsilon = max(self.final_epsilon, self.epsilon-self.epsilon_decay)

# hyper parameters
learning_rate = 0.001
n_episodes = 1000000
initial_epsilon = 1.0
epsilon_decay = initial_epsilon / (n_episodes/2)
final_epsilon = 0.1

agent = BlackjackAgent( learning_rate=learning_rate, initial_epsilon=initial_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)


def print_obs(observation):
    return 
    player_sum, dealer_faceup, player_usable_ace = observation
    print('==============================')
    print(f'player_sum={player_sum}')
    print(f'player_usable_ace={player_usable_ace}')
    print(f'dealer_faceup={dealer_faceup}')


env = gym.wrappers.RecordEpisodeStatistics(env)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    print_obs(obs)

    while not done:
      action = agent.get_action(obs)
      #print(f'action={action}')

      next_obs, reward, terminated, truncated, info = env.step(action)
      print_obs(next_obs)
      #print(f'reward={reward}')
      #print(f'terminated={terminated}')
      #print(f'truncated={truncated}')

      #
      agent.update(obs, action, reward, terminated, next_obs)
      #frame = env.render()
      #plt.imshow(frame)
      #plt.show()

      #
      done = terminated or truncated
      obs = next_obs
    
    agent.decay_epsilon()



wins, losses, draws = 0, 0, 0
for _ in range(1000):
    obs, info = env.reset()
    done = False
    while not done:
        action = np.argmax(agent.q_values[obs])  # Exploit learned policy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1
print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

window = 1000  # Set the window size for smoothing
smooth_error = np.convolve(agent.training_error, np.ones(window) / window, mode='valid')

plt.plot(smooth_error)
plt.xlabel("Training steps")
plt.ylabel("Smoothed Temporal Difference Error")
plt.title("Training Error (Smoothed) Over Time")
plt.show(block=True)

print('done')