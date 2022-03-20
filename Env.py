import random
import numpy as np
import gym
from statistics import mean

num_packets = 10000  # 一共生成多少个包
num_load_balancers = 4  # 均衡器的数量
num_servers = 15  # 服务器的数量
mean_t = 0.004
std = 0.002


class Packet:
    def __init__(self, ip, time_received, processing_time):
        self.ip = ip
        self.time_received = time_received  # 每个包只保留被服务器接收的时间
        self.processing_time = processing_time  # 每个包的处理时间
        self.waiting = False
        self.server_ip=None


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

    def reset(self):
        self.queue.clear()  # 清空请求队列

    def add_packet(self, packet):
        packet.server_ip=self.ip
        self.queue.append(packet)

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
        return len(self.queue)/(num_packets / num_servers), self.get_mean()/mean_t, self.get_std()/std, self.get_percentile()/mean_t

class NetworkEnv(gym.Env):
    def __init__(self, num_packets=0, num_servers=0, num_balancers=0, collaborative=True, server_velocity=None, gamma=0.9, num_clients=50):

        assert num_servers == len(server_velocity)

        self.packets = [Packet(ip=random.randint(0, num_clients - 1), time_received=random.random(), processing_time=np.random.exponential(0.0004)) for i in
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

    def step(self, action_n):
        info = {}
        aa = action_n.tolist()
        a = []
        index = 0
        alpha = []
        for i in aa:
            ssum = sum(i)
            i = [(p / ssum) for p in i]
            alpha.append(i)
            a.append(np.random.choice(np.arange(0, len(self.servers)), p=i))
        action_n = a
        packet = self.packets[self.index: self.index + self.n]
        packet.sort(key=lambda x: x.time_received, reverse=False)

        for i in range(len(packet)):
            self.agents[i].distribute(self.servers[action_n[i]], packet[i])

        self.index += self.n

        done = self.index >= len(self.packets)

        temp = []
        for s in self.servers:
            l, m, s, n = s.get_observation()
            temp.append(l)
            temp.append(m)
            temp.append(s)
            temp.append(n)
        obs = [temp] * self.n

        new_loads = [self.get_stochastic_load(packets=packet, j=j, alpha=aa) for j in range(len(self.servers))]
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

        return obs, [reward] * self.n, done, info

    def reset(self):
        self.packets = [Packet(ip=random.randint(0, self.num_clients - 1), time_received=random.random(), processing_time=np.random.exponential(0.0004)) for i in
                        range(num_packets)]
        self.packets.sort(key=lambda x: x.time_received, reverse=False)
        self.time = 0  # 当前的时刻
        self.index = 0
        self.t = 0
        self.loads = None
        for s in self.servers:
            s.reset()
        temp = []
        for s in self.servers:
            l, m, s, n = s.get_observation()
            temp.append(l)
            temp.append(m)
            temp.append(s)
            temp.append(n)
        obs = [temp] * self.n
        return obs

    def get_stochastic_load(self, packets, j, alpha):
        load_sum = 0
        workload = []
        for lb_index in range(self.n):
            temp = []
            for p in packets:
                if p.ip % self.n == lb_index:
                    temp.append(p.processing_time)
            workload.append(temp)
        for lb_index in range(self.n):
            load_sum += sum(workload[lb_index]) * alpha[lb_index][j]
        return load_sum / self.servers[j].velocity

    def get_deterministic_load(self, packets, j):
        load_sum = 0
        workload = []
        for lb_index in range(self.n):
            temp = []
            for p in packets:
                if p.ip % self.n == lb_index:
                    temp.append(p) # 这个balancer发的包的集合
            workload.append[temp] # 记录哪一个balancer发的这个包
        
        for lb_index in range(self.n):
            for p in workload[lb_index]:
                if p.server_ip == j:
                    load_sum += p.processing_time
        return load_sum / self.servers[j].velocity

    def fairness(self, loads):
        product = 1
        maximum = max(loads)
        if maximum == 0:
            return 0
        for j in range(len(loads)):
            product *= (loads[j] / maximum)
        return product


# if __name__ == '__main__':
    # env = NetworkEnv(num_packets=num_packets, num_servers=num_servers, num_balancers=num_load_balancers,
    #                  collaborative=True, server_velocity=[1]*20)
    # episodes = 10
    # for e in range(1, episodes + 1):
    #     state = env.reset()
    #     done = False
    #     score = 0  # this is the return
    #
    #     while not done:
    #         action_n = th.Tensor([ [0.05]*20 for i in range(env.n)])
    #         obs, reward, done, info = env.step(action_n)
    #         score += reward[0]
    #
    #     print('Episode: {} Score: {}'.format(e, score))