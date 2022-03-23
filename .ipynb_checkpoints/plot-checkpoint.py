import matplotlib.pyplot as plt
import numpy as np
record = open("./reward_record.txt", 'r')

reward = []
for line in record.readlines():
    l = line.split(' ')
    l[-1] = l[-1].replace('\n', '')
    reward.append(l[-1])
    
plt.plot(np.arange(3001), reward)
plt.show()