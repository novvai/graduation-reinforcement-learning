

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np


ENV_NAME = 'RocketLander-v0'
# env = gym.make('RocketLander-v0')
# env = gym.make('Pendulum-v0')
env = gym.make(ENV_NAME)
HIDDEN_LAYERS   = 1024
device = 'cpu'

num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
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

def test_env(env, model, device, deterministic=True, n_tests=10):
    total_rewards, rewards = [], []
    for nt in range(n_tests):
        done = False
        state = env.reset()
        total_reward = 0
        step = 0
        
        while not done:
            step+=1
            state = T.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            
            action = dist.mean.detach() if deterministic \
                else dist.sample()

            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            state = next_state
            total_reward += reward
            print(step, reward)
            env.render()
        total_rewards.append(total_reward)
        rewards.append(reward)
    return total_rewards, rewards
agent = ActorCriticNet(input_dims=num_inputs, layer_1_dims=HIDDEN_LAYERS, action_dims=num_outputs).to(device)

agent.load_state_dict(T.load('./checkpoint_1024'))

test_env(env,agent,device=device, deterministic=True, n_tests=100)
