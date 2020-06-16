import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as FF
import numpy as np

from ppo import ActorCriticNet
from utils import compute_gae, normalize, batch_iter, loginto

np.random.seed(42)
torch.manual_seed(42)

# ENV_NAME = 'LunarLanderContinuous-v2'
ENV_NAME = 'RocketLander-v0'
# env = gym.make('RocketLander-v0')
# env = gym.make('Pendulum-v0')
env = gym.make(ENV_NAME)
test_env_data = gym.make(ENV_NAME)
state = env.reset()

device = 'cpu'

LR              = 4e-5
GAMMA           = .99
NUM_ENVS        = 1
PPO_STEPS       = 4096
BATCH_SIZE      = 128
A_LAMBDA        = .95
PPO_EPOCHS      = 12
TEST_EPOCHS     = 10
PPO_EPSILON     = .2
ENTROPY_BETA    = 2.5e-3
HIDDEN_LAYERS   = 1024
CRITIC_DISCOUNT = .5


num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

agent = ActorCriticNet(input_dims=num_inputs, layer_1_dims=HIDDEN_LAYERS, action_dims=num_outputs).to(device)
print(agent)
print(device)

optimizer = optim.Adam(agent.parameters(), lr=LR)

state = env.reset()
agent.load_state_dict(torch.load('./checkpoint_1024'))

def test_env(env, model, device, deterministic=True, n_tests=10, rend=False):
    total_rewards, rewards = [], []
    for nt in range(n_tests):
        done = False
        state = env.reset()
        total_reward = 0
        
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            action = dist.mean.detach().cpu().numpy()[0] if deterministic \
                else dist.sample().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if rend:
                env.render()
        total_rewards.append(total_reward)
        rewards.append(reward)

    return total_rewards, rewards

def update_policy(states, actions, log_probs, returns, advantages, clip_param=.2):
    for _ in range(PPO_EPOCHS):
        for state, action, old_log_probs, return_, advantage in batch_iter(states, actions, log_probs, returns, advantages, BATCH_SIZE):
            dist, value = agent(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs.to(device)).exp()

            surr1 = ratio * advantage.to(device)
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0+clip_param) * advantage.to(device)

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_.to(device) - value).pow(2).mean()

            loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


ep = 124616
last_reward = -np.inf
should_render = False
for _ in range(1_000_000_000):

    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    dones = []
    rep = 0
    for _ in range(PPO_STEPS):
        rep+=1
        state = torch.tensor(state, dtype=torch.float).to(device)

        dist, value = agent(state)
        action = dist.sample()
        
        next_state, reward, done, _ = env.step(action.cpu().numpy())

        action_log_prob = dist.log_prob(action)

        log_probs.append(action_log_prob)
        values.append(value)
        rewards.append(torch.tensor(reward, dtype=torch.float).to(device))
        dones.append(torch.tensor(done, dtype=torch.float).to(device))

        states.append(state)
        actions.append(action)

        state = next_state
        
        if done :
            rep=0
            print(f'Reward for PPO ep {ep}: {reward}')
            state = env.reset()

    print(f'OPTIMIZING PPO ep {ep}')
    ep+=1
    next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    _, next_value = agent(next_state)

    returns = compute_gae(next_value, rewards, values, dones, A_LAMBDA, GAMMA)
    
    

    returns = torch.tensor(returns, dtype=torch.float).unsqueeze(-1).detach()
    log_probs = torch.stack(log_probs).detach()
    values = torch.tensor(values, dtype=torch.float).unsqueeze(-1).detach()
    states = torch.stack(states)
    actions = torch.stack(actions)
    # print(log_probs)
    advantages = returns - values
    advantages = normalize(advantages)

    update_policy(states, actions, log_probs, returns, advantages, clip_param=PPO_EPSILON)
    if ep % 10 == 0:
        torch.save(agent.state_dict(), './checkpoint_1024')
    if ep % 5 == 0:
        total_test_rewards, last_test_rewards = test_env(test_env_data, agent, device, should_render)
        if last_reward < np.mean(last_test_rewards):
            last_reward = np.mean(last_test_rewards)
            torch.save(agent.state_dict(), './best-rew-checkpoint_1024')
            
        print(f'Reward for ep {ep}: {total_test_rewards} : --LAST-- {last_test_rewards} --AVG_RUN-- {np.mean(last_test_rewards)}')
        print(f'--BEST-- {last_reward}')
        loginto('./data_stats_1024.log', f'{ep}, {np.mean(last_test_rewards)} \n')
        if np.mean(last_test_rewards)>0:
            should_render = True
