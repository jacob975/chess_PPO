from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_chess import GymChessEnv
from callbacks import SetAdversaryCallback
from custom_policy import CustomCNN, TransformerModel
import torch as th

n_env = 4
batch_size = 2048
n_epochs = 5
clip_range = 0.2
nlayers = 1
gamma = 0.99
dropout = 0.4
model_path = "transformer-nlayer1-batch2k-clip02-dropout04-env4-epoch5-ppo-gamma099"
adversary_path = model_path

#env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
env = GymChessEnv()
env = make_vec_env(GymChessEnv, n_envs=n_env, seed=0, vec_env_cls=DummyVecEnv)

policy_kwargs = dict(
    features_extractor_class=TransformerModel,
    features_extractor_kwargs=dict(features_dim=4672, dropout=dropout, nlayers=nlayers),
)

# Agent model
try:
    model = MaskablePPO.load(
        model_path+"/last_model", env=env, gamma=gamma, verbose=1,
        n_epochs=n_epochs,
        batch_size=batch_size,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./maskableppo_chess_tensorboard/",
        device="cuda",
    )
except:
    model = MaskablePPO(
        "MlpPolicy", env, gamma=gamma, seed=None, verbose=1,
        n_epochs=n_epochs,
        batch_size=batch_size,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./maskableppo_chess_tensorboard/",
        device="cuda",
    )

# Adversary model
try: adversary = MaskablePPO.load(adversary_path+"/adversary_model", env=env, verbose=1)
except:
    adversary = MaskablePPO(
        "MlpPolicy", env, gamma=0.99, seed=None, verbose=1,
        policy_kwargs=policy_kwargs,
        device="cuda"
    )

adversary.policy.features_extractor.training = False

callback = SetAdversaryCallback(
    update_freq=1024*n_env, 
    adversary=adversary, 
    model_name=model_path+"/last_model", 
    adversary_name=adversary_path+"/adversary_model",
)

model.learn(
    1e7, callback=callback,
    tb_log_name=model_path,
    reset_num_timesteps=False # Important to keep the same number of timesteps
)