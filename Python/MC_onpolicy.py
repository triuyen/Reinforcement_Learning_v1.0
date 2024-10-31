import gym
import numpy as np
from collections import defaultdict

# Create the environment
class MonteCarlo_onpol():
  def __init__(self, env, epsilon):
    self.env = env
    self.epsilon = epsilon


  # Hyperparameters
  epsilon = 0.1  # for epsilon-greedy policy
  gamma = 0.9    # discount factor
  alpha = 0.02   # learning rate
  num_episodes = 50000  # Number of episodes to run
  eval_interval = 1000

  # Initialize action-value function (Q) as a dictionary
  Q = defaultdict(lambda: np.zeros(self.env.action_space.n))

  # Function to create epsilon-greedy policy
  def epsilon_greedy_policy(self,state, epsilon):
      # If random number is below epsilon, select a random action (exploration)
      if np.random.rand() < epsilon:
          return self.env.action_space.sample()
      # Otherwise, select the action with the highest Q-value (exploitation)
      else:
          return np.argmax(Q[state])

  # Monte Carlo On-Policy method
  for episode in range(num_episodes):
      state = self.env.reset()  # Initialize the first state of the episode
      episode_history = []  # Store state-action-reward tuples for this episode
      # Generate an episode following epsilon-greedy policy
      done = False
      while not done:
          # Choose action based on epsilon-greedy policy
          action = epsilon_greedy_policy(state, epsilon)
          # Take action in the environment
          next_state, reward, done, info = env.step(action)
          # Store the experience (state, action, reward)
          episode_history.append((state, action, reward))
          # Move to the next state
          state = next_state

      # Calculate returns and update Q-values
      G = 0  # Initialize return
      for state, action, reward in reversed(episode_history):
          G = reward + gamma * G  # Calculate return Gt

          # Update Q(s, a) using incremental mean (avoids needing to store returns)
          Q[state][action] = Q[state][action] + alpha * (G - Q[state][action])

  # Extract the final policy (deterministic)
  policy = {state: np.argmax(actions) for state, actions in Q.items()}

  # Optional: Evaluate the learned policy
  def evaluate_policy(self, policy, num_eval_episodes=10000):
      wins = 0
      for _ in range(num_eval_episodes):
          state = env.reset()
          done = False
          while not done:
              action = policy.get(state, env.action_space.sample())  # Use learned policy
              state, reward, done, _ = env.step(action)
          if reward > 0:
              wins += 1
      win_rate = wins / num_eval_episodes
      return win_rate

  #Evaluate the performance of the learned policy
  test = self.evaluate_policy(policy)
  print(test)
  win_rates = []
  #print(f"Win rate after training: {win_rate * 100:.2f}%")