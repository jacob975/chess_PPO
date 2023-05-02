import torch as th
import torch.nn as nn
import numpy as np
import math
from gym import spaces
from torchvision import models
from torchsummary import summary
from torchvision.models import ResNet18_Weights # TODO: Test this model later.

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def customed_resnet(n_input_channels, feature_channels):
    resnet = models.resnet18(weights=None) # None is better than 'imagenet'
    #resnet = models.resnet34(weights=None)
    resnet.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Keep the feature map size the same as the input
    # 1. Remove all maxpooling in resnet18
    resnet.maxpool = nn.Identity()
    # 2. Remove all strides in resnet18
    for m in resnet.modules():
        if isinstance(m, nn.Conv2d):
            m.stride = (1, 1)
    # Remove AvgPool2d and replace it with a Conv2d
    resnet.avgpool = nn.Conv2d(512, feature_channels, kernel_size=3, stride=1, padding='same')
    # Remove the last fully connected layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    return resnet

def customed_resnext(n_input_channels, feature_channels):
    resnext = models.resnext50_32x4d(weights=None)
    resnext.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Keep the feature map size the same as the input
    # 1. Remove all maxpooling in resnet18
    resnext.maxpool = nn.Identity()
    # 2. Remove all strides in resnet18
    for m in resnext.modules():
        if isinstance(m, nn.Conv2d):
            m.stride = (1, 1)
    # Remove AvgPool2d and replace it with a Conv2d
    resnext.avgpool = nn.Conv2d(2048, feature_channels, kernel_size=3, stride=1, padding='same')
    # Remove the last fully connected layer
    resnext = nn.Sequential(*list(resnext.children())[:-1])
    return resnext

def vanilla_cnn(n_input_channels, feature_channels):
    return nn.Sequential(
        nn.Conv2d(n_input_channels,32, kernel_size=3, stride=1, padding='same'),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
        nn.ReLU(),
        nn.Conv2d(64, feature_channels, kernel_size=3, stride=1, padding='same'),
    )

def vanilla_cnn_2(n_input_channels, feature_channels):
    return nn.Sequential(
        nn.Conv2d(n_input_channels,32, kernel_size=3, stride=1, padding='same'),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.0),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
        nn.BatchNorm2d(128),
        nn.Dropout2d(0.0),
        nn.ReLU(),
        nn.Conv2d(128, feature_channels, kernel_size=3, stride=1, padding='same'),
    )

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, training: bool = True):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        feature_channels = features_dim // (observation_space.shape[1] * observation_space.shape[2])
        self.kernel = vanilla_cnn_2(n_input_channels, feature_channels)
        #self.kernel = customed_resnet(n_input_channels, feature_channels)
        #self.kernel = customed_resnext(n_input_channels, feature_channels)
        self.training = training
        self.temp = 1
        self.flatten = nn.Flatten()
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.kernel(x)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.flatten(x)
        return x
    
class TransformerModel(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(
            self, 
            observation_space: spaces.Box, 
            features_dim:int=256, 
            d_model:int=512, 
            nhead:int=4,
            d_hid:int=2048, 
            nlayers:int=4,
            dropout:float=0.1,
        ):
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0] # Input channels
        n_output_channels = features_dim // (observation_space.shape[1] * observation_space.shape[2]) # Output channels
        self.model_type = 'Transformer'
        self.emb = nn.Linear(n_input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_output_channels)
        self.flatten = nn.Flatten()

    def forward(self, x:th.Tensor) -> th.Tensor:
        # Reshape the input from (B, C, H, W) to (H*W, B, C)
        x = x.permute(2, 3, 0, 1) # (H, W, B, C)
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3]) # (H*W, B, C)
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x) # (H*W, B, C)
        x = x.permute(1, 0, 2) # (B, H*W, C)
        x = self.flatten(x) # (B, H*W*C)
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = th.sin(position * div_term)
        pe[:, 0, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Custom CNN for supervised learning
# This model should be the same as CustomCNN
class SupervisedCNN(th.nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, normalize: bool = True, activation_fn: th.nn.Module = th.nn.Sigmoid()):
        super().__init__()
        n_input_channels = observation_space.shape[0]
        feature_channels = features_dim // (observation_space.shape[1] * observation_space.shape[2])
        self.kernel = vanilla_cnn_2(n_input_channels, feature_channels)
        #self.kernel = customed_resnet(n_input_channels, feature_channels)
        #self.kernel = customed_resnext(n_input_channels, feature_channels)
        self.flatten = nn.Flatten()
        self.normalize = normalize
        self.activation_fn = activation_fn

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.kernel(x)
        x = x.permute(0, 2, 3, 1)
        x = self.flatten(x)
        if self.normalize:
            x = x / th.norm(x, dim=1, keepdim=True)
        x = self.activation_fn(x)
        return x


# Unit test
if __name__ == '__main__':
    model = CustomCNN(
        observation_space=spaces.Box(low=0, high=1, shape=(111, 8, 8), dtype=np.float32), 
        features_dim=4672
    )
    # Activate cuda
    model = model.cuda()
    print(model)
    summary(model, (111, 8, 8))

    supervised_model = SupervisedCNN(
        observation_space=spaces.Box(low=0, high=1, shape=(111, 8, 8), dtype=np.float32),
        features_dim=4672,
        normalize=False,
        activation_fn=th.nn.Sigmoid(),
    )
    supervised_model = supervised_model.cuda()
    summary(supervised_model, (111, 8, 8))

    # Assign the weights of the custom CNN to the supervised CNN
    #supervised_model.load_state_dict(th.load("./models/supervised_model.pth"))
    supervised_model.kernel.load_state_dict(model.kernel.state_dict())

    # Test the transformer model
    transformer_model = TransformerModel(
        observation_space=spaces.Box(low=0, high=1, shape=(111, 8, 8), dtype=np.float32),
        features_dim=4672,
        d_model=512,
        nhead=4,
        d_hid=2048,
        nlayers=4,
        dropout=0.5,
    )
    transformer_model = transformer_model.cuda()
    print(transformer_model)
    transformer_model(th.randn(2, 111, 8, 8).cuda())
    #summary(transformer_model, (111, 8, 8)) # torchsummary bug
