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
high = np.inf * np.ones(4 * 8)
print(high.shape)
low = -high
action_space = MultiDiscrete([8, 8, 8, 8])
observation_space = Box(shape=(32,), low=low, high=high)

print(action_space.sample())
print(observation_space.sample())
