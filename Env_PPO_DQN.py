import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
from copy import deepcopy
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box, MultiDiscrete
import matplotlib.pyplot as plt
from statistics import mean

num_packets = 1000  # 一共生成多少个包
num_load_balancers = 4  # 均衡器的数量
num_servers = 8  # 服务器的数量


class Memory:
    def __init__(self, capacity=num_packets, lbs=num_load_balancers, servers=num_servers):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        '''
        self.counts = [{} for i in range(lbs)]
        for d in self.counts:
            for i in range(servers):
                d[i] = 0
        '''

    def push(self, workloads, alpha):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = [workloads, alpha]
        self.position = (self.position + 1) % self.capacity
        '''
        # print(type(alpha))
        # print(alpha)
        for i in range(len(alpha)):
            if alpha[i] in self.counts[i]:
                self.counts[i][alpha[i]] += 1
            else:
                self.counts[i][alpha[i]] = 1
        '''

    def get_memory(self, index):
        workload, alpha = self.memory[index]
        return workload, alpha

    def get_frequency(self):
        # print(self.counts)
        frequency = []
        for d in self.counts:
            temp = []
            temp_sum = sum(d.values())
            for k in d:
                temp.append(d[k] / temp_sum)
            frequency.append(temp)
        # print(frequency)
        return frequency

    def __len__(self):
        return len(self.memory)


class Packet:
    def __init__(self, ip, time_received, processing_time):
        self.ip = ip
        self.time_received = time_received  # 每个包只保留被服务器接收的时间
        self.processing_time = processing_time  # 每个包的处理时间
        self.waiting = False


# 代表均衡器
class LoadBalancer:
    def __init__(self, ip):
        self.ip = ip

    def distribute(self, server, packet):
        return server.add_packet(packet)


# 代表服务器，假设服务器用的是FCFS，并且是non-preemptive
class Server:
    def __init__(self, ip, velocity):
        self.ip = ip
        self.queue = []  # 服务器的请求队列
        self.velocity = velocity
        self.total_load = 0

    def reset(self):
        self.queue.clear()  # 清空请求队列
        self.total_load = 0

    def add_packet(self, packet):
        self.queue.append(packet)
        self.total_load += packet.processing_time

    def get_total_load(self):
        return self.total_load

    def get_mean(self):
        if len(self.queue) == 0:
            return 0

        return mean([p.processing_time for p in self.queue])

    def get_std(self):
        if len(self.queue) == 0:
            return 0

        l = [p.processing_time for p in self.queue]
        mean = sum(l) / len(l)
        variance = sum([((x - mean) ** 2) for x in l]) / len(l)
        res = variance ** 0.5
        return res

    def get_percentile(self, percentile=90):
        if len(self.queue) == 0:
            return 0

        a = np.array([p.processing_time for p in self.queue])
        return np.percentile(a, percentile)

    def get_observation(self):
        return len(self.queue) / (
                num_packets / num_servers), self.get_mean() / 0.004 / self.velocity, self.get_std() / 0.004 / self.velocity, self.get_percentile() / 0.004 / self.velocity


class NetworkEnv(gym.Env):
    def __init__(self, num_packets=0, num_servers=0, num_balancers=0, collaborative=True, server_velocity=None,
                 gamma=0.9, num_clients=50):

        assert num_servers == len(server_velocity)

        self.packets = [Packet(ip=random.randint(0, num_clients - 1), time_received=random.random(),
                               processing_time=np.random.exponential(scale=0.004)) for i in
                        range(num_packets)]
        self.packets.sort(key=lambda x: x.time_received, reverse=False)

        self.server_velocity = server_velocity  # 记录所有server的最大的吞入量

        self.n = num_balancers  # agents的数量，也就是均衡器的数量
        self.shared_reward = collaborative  # 是否是合作模式
        self.time = 0  # 当前的时刻
        self.agents = [LoadBalancer(i) for i in range(num_balancers)]
        self.servers = [Server(i, self.server_velocity[i]) for i in range(num_servers)]
        self.waiting_packets = []
        self.index = 0
        self.t = 0
        self.gamma = gamma
        self.loads = None
        self.num_clients = num_clients
        self.memory = Memory()
        high = np.inf * np.ones(4 * num_servers)
        low = -high
        self.action_space = MultiDiscrete([8, 8, 8, 8])
        self.observation_space = Box(shape=(4 * num_servers,), low=low, high=high)

    def step(self, action_n):
        print(action_n)
        info = {}
        index = 0
        alpha = []
        '''
        for i in aa:
            ssum = sum(i)
            i = [(p / ssum) for p in i]
            alpha.append(i)
            a.append(np.random.choice(np.arange(0, len(self.servers)), p=i))
        action_n = a
        '''
        packet = self.packets[self.index: self.index + self.n]
        packet.sort(key=lambda x: x.time_received, reverse=False)
        for i in range(len(packet)):
            self.agents[packet[i].ip % self.n].distribute(self.servers[action_n[i]], packet[i])

        self.index += self.n

        done = self.index >= len(self.packets)

        temp = []
        for s in self.servers:
            l, m, s, n = s.get_observation()
            temp.append(l)
            temp.append(m)
            temp.append(s)
            temp.append(n)
        obs = temp

        workload = []
        for lb_index in range(self.n):
            temp = []
            for p in packet:
                if p.ip % self.n == lb_index:
                    temp.append(p.processing_time)
            workload.append(temp)

        self.memory.push(workloads=workload, alpha=alpha)

        new_loads = [self.get_stochastic_load(workload=workload, j=j, alpha=alpha) for j in range(len(self.servers))]

        # new_loads = [self.get_deterministic_load(j=j) for j in range(len(self.servers))]
        if self.t == 0:
            reward = self.fairness(loads=new_loads)
            self.loads = new_loads
        else:
            loads = []
            for i in range(len(self.servers)):
                loads.append((1 - self.gamma) * self.loads[i] + self.gamma * new_loads[i])
            reward = self.fairness(loads=loads)
            self.loads = new_loads
        self.t += 1
        return np.array(obs), [reward] * self.n, done, info

    def reset(self):
        self.packets = [Packet(ip=random.randint(0, self.num_clients - 1), time_received=random.random(),
                               processing_time=np.random.exponential(scale=0.004)) for i in
                        range(num_packets)]
        self.packets.sort(key=lambda x: x.time_received, reverse=False)
        self.time = 0  # 当前的时刻
        self.index = 0
        self.t = 0
        self.loads = None
        self.memory = Memory()
        for s in self.servers:
            s.reset()
        temp = []
        for s in self.servers:
            l, m, s, n = s.get_observation()
            temp.append(l)
            temp.append(m)
            temp.append(s)
            temp.append(n)
        obs =  temp
        return np.array(obs)

    def get_deterministic_load(self, j):
        load_sum = self.servers[j].get_total_load()
        return load_sum / self.servers[j].velocity

    def get_stochastic_load(self, workload, j, alpha):
        load_sum = 0

        for lb_index in range(self.n):
            load_sum += sum(workload[lb_index]) * alpha[lb_index][j]

        ''' 从t0开始累积
        for t in range(len(self.memory)):
            temp_workload, temp_alpha = self.memory.get_memory(index=t)
            for lb_index in range(self.n):
                load_sum += sum(temp_workload[lb_index]) * temp_alpha[lb_index][j]
        '''

        return load_sum / self.servers[j].velocity

    def fairness(self, loads):
        product = 1
        maximum = max(loads)
        if maximum == 0:
            return 0
        for j in range(len(loads)):
            product *= (loads[j] / maximum)
        return product

    def calculate_alpha(self):
        alpha = []


if __name__ == '__main__':
    env = NetworkEnv(num_packets=num_packets, num_servers=num_servers, num_balancers=num_load_balancers,
                     collaborative=True, server_velocity=[1, 1, 1, 1, 1, 1, 1, 1])
    episodes = 10
    for e in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0  # this is the return

        while not done:
            action_n = th.Tensor([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125] for i in range(env.n)])
            obs, reward, done, info = env.step(action_n)
            score += sum(reward)

        print('Episode: {} Score: {}'.format(e, score))
