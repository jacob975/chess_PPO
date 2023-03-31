from stable_baselines3 import ppo
import time
import numpy as np
from gym_chess import GymChessEnv

# Create the environment
env = GymChessEnv()
env.reset()
print(env.render())

# Create the agent
model = ppo.PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)