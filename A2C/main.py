import gym, math, random, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FF
import numpy as np
from actor_critic import ActorCritic

env = gym.make('RocketLander-v0')

actor_critic_agent = ActorCritic()

state = env.reset()
GAMMA = .99
lr = 0.0001

optimizer = optim.Adam(params=actor_critic_agent.parameters(), lr=lr)
running_score = 0

eps = 0.01
start_eps = eps
eps_final = .01
eps_decay = 50_000

render = False

ep = 1

# load state 
# actor_critic_agent.load_state_dict(torch.load('./checkpoints/checkpoint-700000.pth'))
# load state
running_actions = []
for i in range(0,1_000_000_000):
    t_state = torch.tensor(state, dtype=torch.float)

    probabilities, _ = actor_critic_agent(t_state)

    probabilities = FF.softmax(probabilities, dim=0)

    actions_prob = torch.distributions.Categorical(probabilities)

    eps = eps_final + (start_eps - eps_final) * math.exp(-1. * i / eps_decay)
    if eps > random.random():
        action = env.action_space.sample()
        action = torch.tensor(action, dtype=torch.long)
    else:
        action = actions_prob.sample()

    log_probs = actions_prob.log_prob(action)

    next_state, reward, done, _ = env.step(action.item())
    if done is True:
        reward += -((next_state[2]/50)**2)
        reward += -((next_state[8]/50)**2)
    else:
        reward += -((next_state[2] / 150) ** 2)
        reward += -((next_state[8] / 100) ** 2)

    running_score += reward
    running_actions.append(action.item())
    ## Learning
    optimizer.zero_grad()

    t_next_state = torch.tensor(next_state, dtype=torch.float)

    _, critic_value = actor_critic_agent(t_state)
    _, next_critic_value = actor_critic_agent(t_next_state)

    t_reward = torch.tensor(reward, dtype=torch.float)

    # TD Loss - Temporal Difference
    delta = t_reward + (1-int(done)) * (GAMMA * next_critic_value) - critic_value

    actor_loss = -log_probs * delta
    critic_loss = delta ** 2
    loss_value = (actor_loss + critic_loss)
    loss_value.backward()
    optimizer.step()
    if(render):
        env.render()

    if ep%10 == 0:
        render = True
        
    if i % 100_000 == 0:
        torch.save(actor_critic_agent.state_dict(), f'./checkpoints/checkpoint-{i}.pth')
        print("LOSS ", loss_value.item())

    ## Now reset
    if done:
        ep+=1
        render = False
        state = env.reset()
        print(f'Running score (ep {ep}) : {running_score}')
        print(f'Running eps : {eps}')
        print(f'Running actions : {running_actions}')
        running_score = 0
