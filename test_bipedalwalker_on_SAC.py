#!/usr/bin/env python
# coding: utf-8

# In[1]:


from stable_baselines3 import sac
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym
import time
import numpy as np


# In[2]:


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


# In[3]:


env_id = "BipedalWalker-v3"
num_cpu = 4  # Number of processes to use
env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)


# In[4]:


# Create a SAC agent
agent = sac.SAC('MlpPolicy', env, verbose=1)


# In[6]:


# Train on the environment save the best model
agent.learn(
    total_timesteps=1000000, 
    log_interval=10, 
    tb_log_name="SAC_BipedalWalker-v3_100k", 
)


# In[5]:


# Save the model
agent.save("SAC_BipedalWalker-v3_100k")


# In[7]:


# Load the trained agent
agent = sac.SAC.load("SAC_BipedalWalker-v3_100k")


# In[7]:


# Evaluate the agent with animation
obs = env.reset()

for i in range(1000):
    action, _states = agent.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render()
    time.sleep(0.01)

input()
env.close()


# In[ ]:




