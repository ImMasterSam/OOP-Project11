import torch
import torch.nn as nn
import numpy as np

class CNNNetwork(nn.Module):
    """
    CNN Network for Car Racing Agent (PPO).
    Input: (B, 3, 96, 96)
    Output: 
        - mean: (B, action_dim)
        - log_std: (B, action_dim) (Learned parameter)
        - value: (B, 1)
    """
    def __init__(self, action_dim=3):
        super(CNNNetwork, self).__init__()
        
        # Feature Extractor (Nature CNN structure)
        # Input image size is 96x96
        # Input channels: 4 (Stacked Grayscale frames)
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # -> (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (64, 8, 8)
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 64 * 8 * 8 = 4096
        self.fc_shared = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU()
        )
        
        # Actor Head (Mean)
        self.actor_mean = nn.Linear(512, action_dim)
        
        # Actor Head (Log Std) - Independent parameter, improved stability
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic Head (Value)
        self.critic = nn.Linear(512, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Init actor mean with smaller gain to start with small actions
        nn.init.orthogonal_(self.actor_mean.weight, 0.01)

    def forward(self, x):
        # x: (B, 3, 96, 96)
        x = x / 255.0 # Normalize if not already
        feat = self.features(x)
        shared = self.fc_shared(feat)
        
        mean = self.actor_mean(shared)
        
        # Expansion to batch size handled by broadcasting during distribution creation
        # But we return the parameter tensor expanded here for consistency
        log_std = self.actor_logstd.expand_as(mean)
        
        value = self.critic(shared)
        
        return mean, log_std, value

    def get_value(self, x):
        x = x / 255.0
        feat = self.features(x)
        shared = self.fc_shared(feat)
        return self.critic(shared)
