import gym

from stable_baselines3 import A2C

model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=10000)

# Enter the tensorboard with the following command:
# tensorboard --logdir ./a2c_cartpole_tensorboard/