from stable_baselines3 import ppo
from sb3_contrib import MarkablePPO
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
from config import *
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
env = make_vec_env(GymChessEnv, n_envs=n_env, seed=0, vec_env_cls=DummyVecEnv)
model = ppo.PPO(
    "MlpPolicy", env, verbose=1, 
    policy_kwargs=policy_kwargs,
    n_steps=n_steps,
    n_epochs=n_epochs,
    batch_size=n_steps*n_env, # Num of minibatch = 1
    clip_range=clip_range,
    tensorboard_log="./ppo_chess_tensorboard/",
)

# 3. optional: load previous weights
#model = ppo.PPO.load("chess_ppo", env=env, policy_kwargs=policy_kwargs)
summary(model.policy.features_extractor.kernel, (111, 8, 8))
# 4. Load the adversary
adversary = ppo.PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
#adversary = ppo.PPO.load("chess_ppo", env=env, policy_kwargs=policy_kwargs)

# 5. Load the weights of the supervised model
#supervised_model = SupervisedCNN(
#    observation_space=spaces.Box(low=0, high=1, shape=(111, 8, 8), dtype=np.float32),
#    features_dim=4672,
#    normalize=False,
#    activation_fn=th.nn.Sigmoid(),
#)
#supervised_model.load_state_dict(th.load("./models/supervised_model.pth"))
#model.policy.features_extractor.kernel.load_state_dict(supervised_model.kernel.state_dict())
#del supervised_model

print("Training...")
# 4. Train the agent
callback = SetAdversaryCallback(update_freq=2048, adversary=adversary)
model.learn(total_timesteps=1e7, tb_log_name="Default weights initialization", callback=callback, log_interval=10)

# 5. Save the agent
model.save("chess_ppo")