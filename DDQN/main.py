import gym, math, random, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FF
import numpy as np
from collections import deque
from utils import loginto

env = gym.make('RocketLander-v0')

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU()
        )

        self.advantage_estimator = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

        self.value_estimator = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        advantage = self.advantage_estimator(x)
        value = self.value_estimator(x)

        return value + (advantage - advantage.mean())


class ExperienceBuffer:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, size))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


def compute_temporal_difference(experiences, batch_size=32, gamma=.98):
    states, actions, rewards, next_states, dones = experiences.sample(size=batch_size)

    states = torch.from_numpy(np.asarray(states, dtype=np.float)).float()

    actions = torch.from_numpy(np.asarray(actions, dtype=np.long)).long()
    rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float)).float()
    next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float)).float()
    dones = torch.from_numpy(np.asarray(dones, dtype=np.float)).float()

    # print(current_agent(states), actions.unsqueeze(1))
    q_values = current_agent(states).gather(1, actions.unsqueeze(1))
    q_values = q_values.squeeze(1)

    # next_q_values = target_agent(next_states)
    next_q_values = current_agent(next_states)
    max_next_q_values = next_q_values.max(1)[0]

    expected_q_value = rewards + gamma * max_next_q_values * (1 - dones)

    return loss_criterion(q_values, expected_q_value)


def update_net(c, t):
    t.load_state_dict(c.state_dict())


state = env.reset()
print(env.action_space)
print(env.observation_space)

current_agent = Actor()
# target_agent = Actor()
experience = ExperienceBuffer(1_000_000)
BATCH_SIZE = 32
GAMMA = .98
n_actions = env.action_space.n
loss_buffer = []
loss_criterion = nn.MSELoss()
optimizer = optim.Adam(params=current_agent.parameters(), lr=.0001)
start_eps = 0.01

eps = 0.01
eps_final = .01
eps_decay = 10_000
running_rewards = []
avg_per_100 = []

current_agent.load_state_dict(torch.load(f'./no_mod/checkpoint-1000000.pth'))
# target_agent.load_state_dict(current_agent.state_dict())
for i in range(1_137_930, 6_000_000):
    tensor_state = torch.tensor(state, dtype=torch.float)
    eps = eps_final + (start_eps - eps_final) * math.exp(-1. * i / eps_decay)

    if eps > random.random():
        action = env.action_space.sample()
    else:
        actions_distr = current_agent(tensor_state)
        action = np.argmax(actions_distr.detach().numpy())

    if i % 200 == 0:
        print(eps)

    next_state, reward, done, info = env.step(action)
    running_rewards.append(reward)
    experience.add(state, action, reward, next_state, done)

    if len(experience) > BATCH_SIZE:
        loss_value = compute_temporal_difference(experience, BATCH_SIZE, GAMMA)
        loss_buffer.append(loss_value.item())
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    if i % 1_000_000 == 0:
        torch.save(current_agent.state_dict(), f'./no_mod/checkpoint-{i}.pth')
     
    state = next_state
    # env.render()

    if done:
        print(f'Reward for step {i + 1} : {np.sum(running_rewards)}')
        avg_per_100.append(np.sum(running_rewards))
        if len(avg_per_100) == 100:
            loginto('./no_mod/data_log.txt', f'{i + 1},{np.mean(avg_per_100)} \n')
            avg_per_100 = []

        running_rewards = []
        state = env.reset()
