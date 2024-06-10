"""NeRF models.

Contains the various models and sub-models used to train a Neural Radiance Field (NeRF).
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer


class NeRFModel(nn.Module):
    """A single NeRF model.

    A single NeRF model (used for both coarse and fine networks) is made up of an
    8-layer multi-layer perceptron with ReLU activation functions. The input is a
    position and direction (each of which are 3 values), while the output is a
    scalar density and rgb value. The final layer predicting the rgb uses a sigmoid
    activation.
    """

    def __init__(self, position_dim=10, direction_dim=4, hidden_dim=256):
        """NeRF Constructor.

        Args:
            position_dim: the size of the position encoding. Resulting size will be 
                input_size*2*position_dim.
            direction_dim: the size of the direction encoding. Resulting size will be 
                input_size*2*direction_dim.
        """
        super(NeRFModel, self).__init__()
        self.position_dim = position_dim
        self.direction_dim = direction_dim
        # first MLP is a simple multi-layer perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(self.position_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.feature_fn = nn.Sequential(
            nn.Linear(hidden_dim + self.position_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(128, 128),
        )

        self.density_fn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU() # rectified to ensure nonnegative density
        )

        self.rgb_fn = nn.Sequential(
            nn.Linear(hidden_dim + self.direction_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, pos_enc_samples, pos_enc_ray_dir): 
        """Forward pass through a NeRF Model (MLP).

        Args:
            samples: [N x samples x 3] coordinate locations to query the network.
            direc: [N x 3] directions for each ray.
        Returns: 
            density: [N x samples x 1] density predictions.
            rgb: [N x samples x 3] color/rgb predictions.
        """
        # feed forward network
        x_features = self.mlp(pos_enc_samples)
        # concatenate positional encodings again
        x_features = torch.cat((x_features, pos_enc_samples), dim=-1)
        x_features = self.feature_fn(x_features)
        density = self.density_fn(x_features)
        # final rgb predictor
        dim_features = torch.cat((x_features, pos_enc_ray_dir), dim=-1)
        rgb = self.rgb_fn(dim_features)
        return rgb, density
