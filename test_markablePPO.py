from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_chess import GymChessEnv
from callbacks import SetAdversaryCallback
from config import *


#env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
env = GymChessEnv()
env = make_vec_env(GymChessEnv, n_envs=n_env, seed=0, vec_env_cls=DummyVecEnv)

model = MaskablePPO(
    "MlpPolicy", env, gamma=0.4, seed=32, verbose=1,
    n_steps=n_steps,
    n_epochs=n_epochs,
    batch_size=n_steps*n_env, # Num of minibatch = 1
    clip_range=clip_range,
    tensorboard_log="./markableppo_chess_tensorboard/"
)

adversary = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
callback = SetAdversaryCallback(update_freq=2048, adversary=adversary)
model.learn(1e7, callback=callback)

#evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

model.save("ppo_mask")
del model # remove to demonstrate saving and loading