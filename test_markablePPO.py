from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_chess import GymChessEnv
from callbacks import SetAdversaryCallback
from custom_policy import CustomCNN
from config import *
import torch as th


#env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
env = GymChessEnv()
env = make_vec_env(GymChessEnv, n_envs=n_env, seed=0, vec_env_cls=DummyVecEnv)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=4672), # 64*7*7
)

model = MaskablePPO(
    "MlpPolicy", env, gamma=0.4, seed=32, verbose=1,
    #n_steps=n_steps,
    #n_epochs=n_epochs,
    #clip_range=clip_range,
    batch_size=256,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./markableppo_chess_tensorboard/"
)

adversary = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
callback = SetAdversaryCallback(update_freq=2048, adversary=adversary)
model.learn(1e7, callback=callback, tb_log_name="resnet34-batch256-env4-ppo")

model.save("resnet34-batch256-env4-ppo")
del model # remove to demonstrate saving and loading