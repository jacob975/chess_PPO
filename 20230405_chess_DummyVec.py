from stable_baselines3 import ppo
import time
import numpy as np
from gym_chess import GymChessEnv
from custom_policy import CustomCNN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from callbacks import SetAdversaryCallback


# Test the environment
env = GymChessEnv()
env.reset()
print(env.render())

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=4672), # 64*7*7
)

# Create a PPO agent
env = make_vec_env(GymChessEnv, n_envs=4, seed=0, vec_env_cls=DummyVecEnv)
model = ppo.PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
adversary = ppo.PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

print("Training...")
# Train the agent
callback = SetAdversaryCallback(update_freq=1e4, adversary=None)
model.learn(total_timesteps=1e6, tb_log_name="chess_ppo", callback=callback)

# Save the agent
model.save("chess_ppo")