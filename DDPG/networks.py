import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

from priority_experience_replay import PriorityExperienceBuffer
from exploration_noise import OUActionNoise

class CriticNet(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,name, checkpoint_dir = 'tmp/ddpg'):
        super(CriticNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}_ddpg')

        # First Layer
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])

        nn.init.uniform_(self.fc1.weight.data, -f1,f1)
        nn.init.uniform_(self.fc1.bias.data, -f1,f1)
        # Batch normalization Layer
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        # Second Layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])

        nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        nn.init.uniform_(self.fc2.bias.data, -f2,f2)
        # Batch normalization Layer
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        # Action value function
        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = .003
        self.q = nn.Linear(self.fc2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -f3,f3)
        nn.init.uniform_(self.q.bias.data, -f3,f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        state_val = self.fc1(state)
        state_val = self.bn1(state_val)
        state_val = F.relu(state_val)

        state_val = self.fc2(state_val)
        state_val = self.bn2(state_val)

        action_value =  F.relu(self.action_value(action))

        state_action_val =  F.relu(torch.add(state_val, action_value))

        state_action_val = self.q(state_action_val)

        return state_action_val

    def save(self):
        print('.-. Saving .-.')
        torch.save(self.state_dict(), self.checkpoint_file)
    def load(self):
        print('.-. Loading Last Checkpoint .-.')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNet(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions,name, checkpoint_dir = 'tmp/ddpg'):
        super(ActorNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}_ddpg')

        # First Layer
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])

        nn.init.uniform_(self.fc1.weight.data, -f1,f1)
        nn.init.uniform_(self.fc1.bias.data, -f1,f1)
        # Batch normalization Layer
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        # Second Layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])

        nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        nn.init.uniform_(self.fc2.bias.data, -f2,f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = .003

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        nn.init.uniform_(self.mu.weight.data, -f3,f3)
        nn.init.uniform_(self.mu.bias.data, -f3,f3)

        self.optimizer = optim.Adam(params=self.parameters(), lr=alpha)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))
        
        return x

    def save(self):
        print('.-. Saving .-.')
        torch.save(self.state_dict(), self.checkpoint_file)
    def load(self):
        print('.-. Loading Last Checkpoint .-.')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=.99, n_actions=3,
                max_size=1_000_000, layer1_s=400, layer2_s=300, batch_size=64):
        self.gamma =gamma
        self.tau = tau
        self.memory = PriorityExperienceBuffer(max_size)
        self.batch_size = batch_size

        self.actor = ActorNet(alpha, input_dims, layer1_s, layer2_s,
                                n_actions=n_actions, name='AgentActor')

        self.target_actor = ActorNet(alpha, input_dims, layer1_s, layer2_s,
                                n_actions=n_actions, name='TargetAgentActor')

        self.critic = CriticNet(beta, input_dims, layer1_s, layer2_s, n_actions=n_actions, name='AgentCritic')
        self.target_critic = CriticNet(beta, input_dims, layer1_s, layer2_s, n_actions=n_actions, name='TargetAgentCritic')

        self.noise = OUActionNoise(mu= np.zeros(n_actions))
        
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        self.actor.eval()
        obs = torch.tensor(obs, dtype=torch.float).to(self.actor.device)
        mu = self.actor(obs).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)

        self.actor.train()

        return mu_prime.cpu().detach().numpy()
    
    def _getPriority(self, err):
        return (err+.01)**.6

    def remember(self, state, action, reward, new_state, done):
        prob = self._getPriority(0)
        self.memory.add(prob, [state, action, reward, new_state, done])

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        idxs, samples, is_weights = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*samples)

       
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.critic.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.critic.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.critic.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.critic.device)
        states = torch.tensor(states, dtype=torch.float).to(self.critic.device)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor(next_states)
        critic_value_ = self.target_critic(next_states, target_actions)
        critic_value = self.critic(states,actions)

        target = []

        for j in range(self.batch_size):
            target.append(rewards[j] + self.gamma * critic_value_[j] * (1-dones[j]))
        
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)
        #### update tree
        target_errors = (target - critic_value)
        for idx,err in zip(idxs, target_errors.cpu().detach().numpy()):
            self.memory.update_tree(idx, self._getPriority(np.abs(err[0])))
        #### update tree
        is_weights = torch.tensor(is_weights, dtype=torch.float).to(self.critic.device)
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = (is_weights * F.mse_loss(target, critic_value)).mean()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()

        self.actor.optimizer.zero_grad()
        mu = self.actor(states)
        self.actor.train()

        actor_loss = -self.critic(states,mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone()+\
                                        (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone()+\
                                        (1-tau)*target_actor_state_dict[name].clone()
                                        
        self.target_actor.load_state_dict(actor_state_dict)

    def save_model(self):
        self.actor.save()
        self.critic.save()
        self.target_actor.save()
        self.target_critic.save()
    def load_model(self):
        self.actor.load()
        self.critic.load()
        self.target_actor.load()
        self.target_critic.load()


