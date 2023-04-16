from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_chess import GymChessEnv
from callbacks import SetAdversaryCallback
from custom_policy import CustomCNN
import torch as th

n_env = 4
batch_size = 1024
n_epochs = 5


#env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
env = GymChessEnv()
env = make_vec_env(GymChessEnv, n_envs=n_env, seed=0, vec_env_cls=DummyVecEnv)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=4672), # 64*7*7
)

model = MaskablePPO(
    "MlpPolicy", env, gamma=0.4, seed=None, verbose=1,
    n_epochs=n_epochs,
    batch_size=batch_size,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./markableppo_chess_tensorboard/"
)
# Reload the model
model = MaskablePPO.load(
    "best_model", env=env, verbose=1,
    n_epochs=n_epochs,
    batch_size=batch_size,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./markableppo_chess_tensorboard/"
)

adversary = MaskablePPO(
    "MlpPolicy", env, gamma=0.4, seed=None, verbose=1,
    policy_kwargs=policy_kwargs,
)
callback = SetAdversaryCallback(update_freq=2048*n_env, adversary=adversary)
model.learn(
    1e7, callback=callback,
    tb_log_name="resnet18-batch1024-env4-epoch5-ppo", 
    reset_num_timesteps=False # Important to keep the same number of timesteps
)

model.save("resnet18-batch1024-env4-epoch5-ppo")
del model # remove to demonstrate saving and loading