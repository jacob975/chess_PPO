from stable_baselines3 import ppo
import torch
import time
import numpy as np
from gym_chess import GymChessEnv
from gym import spaces
import torch as th
from custom_policy import CustomCNN, SupervisedCNN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from callbacks import SetAdversaryCallback
from torchsummary import summary
# Not to allocate all the memory


# 1. Test the environment
env = GymChessEnv()
env.reset()
print(env.render())

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=4672), # 64*7*7
)

# 2. Create a PPO agent
env = make_vec_env(GymChessEnv, n_envs=6, seed=0, vec_env_cls=DummyVecEnv)
model = ppo.PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
summary(model.policy.features_extractor.kernel, (111, 8, 8))
# 3. Load the adversary
adversary = ppo.PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

# Put agent and adversary in the same device
env.env_method("estimate_winrate", agent = model.predict, adversary = adversary.predict)