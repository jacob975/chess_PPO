import torch as th
import torch.nn as nn
import numpy as np
from gym import spaces
from torchvision import models
from torchsummary import summary

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.resnet18 = models.resnet18(pretrained=True)
        # Set the input channels to the correct number
        self.resnet18.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Set the output features to the correct number
        self.resnet18.fc = nn.Linear(512, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.resnet18(observations)
    
# Unit test
if __name__ == '__main__':
    model = CustomCNN(
        observation_space=spaces.Box(low=0, high=1, shape=(20, 8, 8), dtype=np.float32), 
        features_dim=256
    )
    # Activate cuda
    model = model.cuda()
    summary(model, (20, 8, 8))