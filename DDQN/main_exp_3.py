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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

        self.value_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        advantage = self.advantage_estimator(x)
        value = self.value_estimator(x)

        return value + (advantage - advantage.mean())


def compute_temporal_difference(experiences, batch_size=32, gamma=.98):
    idxs, samples = experiences.sample(batch_size)
    # for item in samples:
    #     print(item[0])
    states, actions, rewards, next_states, dones = zip(*samples)

    states = torch.from_numpy(np.asarray(states, dtype=np.float)).float()

    actions = torch.from_numpy(np.asarray(actions, dtype=np.long)).long()
    rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float)).float()
    next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float)).float()
    dones = torch.from_numpy(np.asarray(dones, dtype=np.float)).float()

    # print(current_agent(states), actions.unsqueeze(1))
    q_values = current_agent(states).gather(1, actions.long().unsqueeze(1))
    q_values = q_values.squeeze(1)

    # next_q_values = current_agent(next_states)
    next_q_values = target_agent(next_states)
    max_next_q_values = next_q_values.max(1)[0]

    expected_q_value = rewards + gamma * max_next_q_values * (1 - dones)
    errors = (q_values - expected_q_value)**2
    for idx,err in zip(idxs, errors):
        experience.update_tree(idx, get_probability(err))

    return FF.mse_loss(q_values, expected_q_value)


def get_probability(err):
    return (err + .1) ** 0.6


def update_net(c, t):
    t.load_state_dict(c.state_dict())


state = env.reset()
print(env.action_space)
print(env.observation_space)

current_agent = Actor()
target_agent = Actor()
# load state 
current_agent.load_state_dict(torch.load('./checkpoints/checkpoint-227573.pth'))
# load state
experience = PriorityExperienceBuffer(1_000_000)
BATCH_SIZE = 64
GAMMA = .98
n_actions = env.action_space.n
loss_buffer = []
optimizer = optim.Adam(params=current_agent.parameters(), lr=0.00001)
start_eps = 0.99

eps = 0.99
eps_final = .01
eps_decay = 1_000
running_rewards = []
target_agent.load_state_dict(current_agent.state_dict())
ep = 1

has_saved, has_copied = False,False

for i in range(227573, 1_000_000_000_000):
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

    if done is True:
        reward += -((next_state[2]/50)**2)
        reward += -((next_state[8]/100)**2)
    else:
        reward += -((next_state[2] / 150) ** 2)
        reward += -((next_state[8] / 200) ** 2)

    running_rewards.append(reward)

    experience.add(get_probability(0), [state, action, reward, next_state, done])

    if len(experience) > BATCH_SIZE:
        loss_value = compute_temporal_difference(experience, BATCH_SIZE, GAMMA)
        loss_buffer.append(loss_value.item())
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if ep % 10 == 0 and has_saved is False:
            has_saved = True
            torch.save(current_agent.state_dict(), f'./checkpoints/checkpoint-{i}.pth')
            print("LOSS ", loss_value.item())

        if ep % 2 == 0 and has_copied is False:
            has_copied = True
            print(f'Coppied at {i+1}')
            target_agent.load_state_dict(current_agent.state_dict())


    state = next_state
    if ep % 10 == 0:
        env.render()
    
    if done:
        has_copied = False
        has_saved = False
        print(f'Reward for step {i + 1}(ep {ep}) : {np.sum(running_rewards)}')
        print(f'State for step {i + 1}(ep {ep}) : {state}')
        ep += 1
        running_rewards = []
        state = env.reset()
