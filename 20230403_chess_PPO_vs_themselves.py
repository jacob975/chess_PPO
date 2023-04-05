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
model = ppo.PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
# Duplicate the model as the adversary
adversary = ppo.PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
env.set_adversary(adversary.predict)

print("Training...")
# Train the agent
for i in range(100):
    model.learn(total_timesteps=1e4, tb_log_name="chess_ppo")
    # Update the adversary using the new model
    adversary.set_parameters(model.get_parameters())

# Save the agent
model.save("chess_ppo")