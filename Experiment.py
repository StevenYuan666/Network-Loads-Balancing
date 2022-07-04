from Env import NetworkEnv
from MADDPG import MADDPG
import numpy as np
import torch as th

num_packets = 2000  # 一共生成多少个包
num_load_balancers = 4  # 均衡器的数量
num_servers = 8  # 服务器的数量
PATH= "./"
model_save_name = "parameter.pt"
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
n_agents = num_load_balancers
n_states = num_servers*4 
n_actions = num_servers
capacity = 50000
batch_size = 512

n_episode = 20000
max_steps = 1000
episodes_before_train = 10

win = None
param = None
f1 = open("hetro2.txt", "a+")
maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

world = NetworkEnv(num_packets=num_packets, num_servers=num_servers, num_balancers=num_load_balancers,
                     collaborative=True, server_velocity=[1]*4+[9]*4)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    done = False
    if i_episode % 10 == 0:
        f1.write('Episode: %d\n' % i_episode)

    while not done:
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward, done, _ = world.step(action)

        if i_episode % 10 == 0:
            f1.write("observation: " + str(obs) + "\n")
            f1.write("action: " + str(action) + "\n")

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        next_obs = obs_

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()

    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)
    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on NetworkEnv\n' +
              'agent=%d' % n_agents +
              'server=%d' % num_servers +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n')

    if i_episode % 100 == 0 and i_episode > 0:
        th.save({
            'epoch': i_episode,
            'acotr_state_dict': [[maddpg.actors[i].state_dict()] for i in range(len(maddpg.actors))],
            'critic_state_dict': [[maddpg.critics[i].state_dict()] for i in range(len(maddpg.critics))],
            'acotr_target_state_dict': [[maddpg.actors_target[i].state_dict()] for i in range(len(maddpg.actors))],
            'critic_target_state_dict': [[maddpg.critics_target[i].state_dict()] for i in range(len(maddpg.critics))],
            'actor_optimizer_state_dict': [[maddpg.actor_optimizer[i].state_dict()] for i in
                                           range(len(maddpg.actor_optimizer))],
            'critic_optimizer_state_dict': [[maddpg.critic_optimizer[i].state_dict()] for i in
                                            range(len(maddpg.critic_optimizer))],
        }, PATH + str(i_episode) + model_save_name)

