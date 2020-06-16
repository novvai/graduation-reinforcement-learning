from networks import Agent

import gym
import numpy as np
from utils import loginto

env = gym.make('RocketLander-v0')

agent = Agent(alpha=.0001, beta=.001, input_dims=[10], tau=.001, env=env,batch_size=32, layer1_s=400,layer2_s=300, n_actions=3, max_size=100_000)
# agent.load_model()
np.random.seed(0)
score_history = []
should_render = False
for i in range(1000000000):
    done = False
    score = 0
    obs = env.reset()
    agent.noise.reset()

    while not done:
        act = agent.choose_action(obs)
        # print(act)
        new_state, reward, done, _ = env.step(act)
        agent.remember(obs, act,reward,new_state,int(done))
        agent.learn()
        if (should_render):
            env.render()
        score += reward
        obs=new_state

    score_history.append(score)
    should_render = False
    if i % 100 == 0:
        should_render = True
        loginto('./data.log', f'{i}, {score}, { "passed" if reward>0.5 else "failed"}')
        agent.save_model()
    print(f'Ep: {i}', 'score %.2f' % score, 'Avg 100 : %.2f'%np.mean(score_history[-100:]))

    