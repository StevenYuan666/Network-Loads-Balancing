import matplotlib.pyplot as plt
import numpy as np
record = open("3_21_homogeneous/3.21_reward.txt", 'r')
baseline = open("baseline_homo_random.txt", 'r')

reward = []
for line in record.readlines():
    l = line.split(' ')
    l[-1] = l[-1].replace('\n', '')
    reward.append(float(l[-1]))

base = []
for line in baseline.readlines():
    l = line.split(' ')
    l[-1] = l[-1].replace('\n', '')
    base.append(float(l[-1]))
    
plt.plot(reward[:2000], label="MADDPG based RL model")
plt.plot(base, label="Random Assignment Baseline")
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title("4 Load Balancers and 8 Homogeneous Servers")
plt.legend()
plt.savefig("Homogeneous.png")
