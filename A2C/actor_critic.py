import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
    
        self.feature_ext = nn.Sequential(
            nn.Linear(10,512),
            nn.ReLU(),
        )

        self.actor_seq = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 7)
        )

        self.critic_seq = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs):
        x = self.feature_ext(obs)

        policy = self.actor_seq(x)
        value = self.critic_seq(x)

        return policy, value