import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
from copy import deepcopy
from torch.optim import Adam
import numpy as np

Experience = namedtuple('Experience', ('states', 'actions', 'next_states', 'rewards'))
scale_reward = 0.1

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation #* n_agent
        act_dim = self.dim_action * n_agent

        # self.FC1 = nn.Linear(obs_dim, 32)
        # self.FC2 = nn.Linear(32+act_dim, 32)
        # self.FC3 = nn.Linear(32, 16)
        # self.FC4 = nn.Linear(16, 1)
        self.C1 = nn.Conv1d(1, 1, kernel_size=4, stride=4)
        self.FC2 = nn.Linear(8 + act_dim, 32)
        self.FC3 = nn.Linear(32, 16)
        self.FC4 = nn.Linear(16, 1)
    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        obs = obs.reshape([obs.size(dim=0), 1, obs.size(dim=1)])
        temp = F.relu(self.C1(obs))
        result = temp.reshape([temp.size()[0], temp.size()[2]])
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.C1 = nn.Conv1d(1, 1, kernel_size=4, stride=4)
        self.FC1 = nn.Linear(8, 32)
        self.FC2 = nn.Linear(32, 32)
        self.FC3 = nn.Linear(32, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        obs = obs.reshape([obs.size(dim=0), 1, obs.size(dim=1)])
        temp = F.relu(self.C1(obs))
        result = temp.reshape([temp.size()[0], temp.size()[2]])
        result = F.relu(self.FC1(result))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)
        return result



def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.0002) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.00002) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [self.actors_target[i](non_final_next_states) for i in range(self.n_agents)]#[self.actors_target[i](non_final_next_states[:, i, :]) for i in range(self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())

            target_Q = th.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_states), #self.n_agents * self.n_states
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch#[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = th.zeros(self.n_agents, self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch.detach()#state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            # '''
            act += th.from_numpy(np.random.randn(self.n_actions) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = th.clamp(act, 0, 1.0)
            # '''
            actions[i, :] = act
        self.steps_done += 1

        return F.softmax(actions)

