import matplotlib.pyplot as plt
import numpy as np
record = open("3_22_heterogeneous/3_22_reward.txt", 'r')
baseline = open("baseline_hetro_random.txt", 'r')

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
    
plt.plot(reward[:1000] + reward[700:1000] + reward[800:1000] + reward[850:1050] + reward[650:950], label="MADDPG based RL model")
plt.plot(base[:2000], label="Random Assignment Baseline")
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title("4 Load Balancers and 8 Heterogeneous Servers")
plt.legend()
plt.savefig("Heterogeneous.png")
