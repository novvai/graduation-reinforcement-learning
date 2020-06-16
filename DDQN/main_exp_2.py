import gym, math, random, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FF
import numpy as np
from collections import deque
from priority_experience_replay import PriorityExperienceBuffer

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
            nn.Linear(128, 7)
        )

        self.value_estimator = nn.Sequential(
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
    idxs, samples = experiences.sample(batch_size)
    # for item in samples:
    #     print(item[0])
    states, actions, rewards, next_states, dones = zip(*samples)

    states = torch.from_numpy(np.asarray(states, dtype=np.float32))

    actions = torch.from_numpy(np.asarray(actions, dtype=np.long))
    rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float32))
    next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float32))
    dones = torch.from_numpy(np.asarray(dones, dtype=np.float32))

    # print(current_agent(states), actions.unsqueeze(1))
    q_values = current_agent(states).gather(1, actions.long().unsqueeze(1))
    q_values = q_values.squeeze(1)

    next_q_values = current_agent(next_states)
    max_next_q_values = next_q_values.max(1)[0]

    expected_q_value = rewards + gamma * max_next_q_values * (1 - dones)
    errors = (q_values - expected_q_value)**2
    for idx,err in zip(idxs, errors):
        experience.update_tree(idx, get_probability(err))

    return FF.mse_loss(q_values, expected_q_value)


def get_probability(err):
    return (err + .01) ** 0.6


def update_net(c, t):
    t.load_state_dict(c.state_dict())


state = env.reset()
print(env.action_space)
print(env.observation_space)

current_agent = Actor()
# target_agent = Actor()
experience = PriorityExperienceBuffer(1_000_000)
BATCH_SIZE = 32
GAMMA = .98
n_actions = env.action_space.n
loss_buffer = []
optimizer = optim.Adam(params=current_agent.parameters())
start_eps = 0.99

eps = 0.99
eps_final = .01
eps_decay = 1_000
running_rewards = []
# target_agent.load_state_dict(current_agent.state_dict())

for i in range(1_000_000_000_000):
    tensor_state = torch.tensor(state, dtype=torch.float32)
    eps = eps_final + (start_eps - eps_final) * math.exp(-1. * i / eps_decay)

    if eps > random.random():
        action = env.action_space.sample()
    else:
        actions_distr = current_agent(tensor_state)
        action = np.argmax(actions_distr.detach().numpy())

    if i % 200 == 0:
        print(eps)

    next_state, reward, done, info = env.step(action)

    ### REMOVE
    if done is True:
        reward += -((next_state[2]/10)**2)
        reward += -((next_state[8]/100)**2)
    else:
        reward += -((next_state[2] / 100) ** 2)
        reward += -((next_state[8] / 200) ** 2)
    ### REMOVE

    running_rewards.append(reward)

    experience.add(get_probability(99), [state, action, reward, next_state, done])

    if len(experience) > BATCH_SIZE:
        loss_value = compute_temporal_difference(experience, BATCH_SIZE, GAMMA)
        loss_buffer.append(loss_value.item())
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if i % 2000 == 0:
            print("LOSS ", loss_value.item())




    state = next_state
    env.render()
    if done:
        print(f'Reward for run {i + 1} : {np.sum(running_rewards)}')
        print(f'State for run {i + 1} : {state}')
        print(f'Rewards for run {i + 1} : {running_rewards}')
        running_rewards = []
        state = env.reset()
