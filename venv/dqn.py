from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random

GAMMA = 0.99                #   Discount rate
BATCH_SIZE = 32         #Number of transitions we are gonna sample from replay buffer
BUFFER_SIZE = 50000         #number of transitions we are gonna store before rewriting old ones
MIN_REPLAY_SIZE = 1000          #minimum number of transitions required in buffer before we calculate the gradients and training
EPSILON_START = 1.0             #starting value of the epsilon(epsilon-greedy for exploration)
EPSILON_END = 0.02              #ending value of the epsilon    
EPSILON_DECAY = 10000           #decay time steps of epsilon
TARGET_UPDATE_FREQ = 1000       #number of steps where we set the target parameters equal to online parameters


class Network(nn.Module):
    def __init__(self,env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features,64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))

    def forward(self,x):
        return self.net(x)

    def act(self,obs):
        # print(obs)
        # print(isinstance(obs, tuple))

        OBS = []
        if isinstance(obs,tuple):
            OBS = obs[0]
        else:
            OBS = obs

        obs_t = torch.as_tensor(OBS, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

env = gym.make('CartPole-v0', render_mode="rgb_array")

replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0],maxlen=100)

episode_reward = 0.0


online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

#Initialize Replay Buffer
obs = env.reset()
for x in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    print(len(env.step(action)))
    new_obs, rew, done, x, y = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()

#Main training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, done, _, x= env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += rew

    if done:
        obs = env.reset()

        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 195:
            while True:
                print(1)
                action = online_net.act(obs)

                print(2)
                obs, _, done, _, _ = env.step(action)
                print(3)
                env.render()

                print(4)
                if done:
                    print(5)
                    env.reset()
                    print(6)

    #Start gradient loop
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0][0] if isinstance(t[0],tuple) else t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    #Compute Targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

    #Compute loss
    q_values = online_net(obses_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    #Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    #Logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Rew', np.mean(rew_buffer))

