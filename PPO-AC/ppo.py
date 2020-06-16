import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCriticNet(nn.Module):

    def __init__(self, input_dims, layer_1_dims, action_dims, std=.0):
        super(ActorCriticNet, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dims, layer_1_dims),
            nn.ReLU(),
            nn.Linear(layer_1_dims, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_dims, layer_1_dims),
            nn.ReLU(),
            nn.Linear(layer_1_dims, action_dims)
        )

        self.log_std = nn.Parameter(T.ones(1, action_dims) * std)

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().squeeze(0).expand_as(mu)
        dist = T.distributions.Normal(mu, std)

        return dist, value