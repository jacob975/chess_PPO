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
env.set_adversary(model.predict)

print("Training...")
# Train the agent
model.learn(total_timesteps=1e6, tb_log_name="chess_ppo", log_interval=10)

# Save the agent
model.save("chess_ppo")