from stable_baselines3 import ppo
import time
import numpy as np
from gym_chess import GymChessEnv
from custom_policy import CustomCNN

# Create the environment
env = GymChessEnv()
env.reset()
print(env.render())

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=4672), # 64*7*7
)

# Load previous saved model if available
try:
    model = ppo.PPO.load("chess_ppo", env=env)
except:
    model = ppo.PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

print("Training...")
# Train the agent
model.learn(total_timesteps=1e6, tb_log_name="chess_ppo", log_interval=10)

# Save the agent
model.save("chess_ppo")