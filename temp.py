from mpi4py import MPI
import gym
import numpy as np
from collections import deque
import DQNagent
import tensorflow as tf
import timeit


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

EPISODES = 2

GANMA = 0.99    # discount rate
EPSILON = 1.0  # exploration rate
#Works with min 0.01
EPSILON_MIN = 0.00
EPSILON_DECAY = 0.995
# works with learning rate 0.001
LEARNING_RATE = 0.001
FILE_NAME = "ann-weights.h5"

env = gym.make('LunarLander-v2')
state_size = 8
action_size = env.action_space.n

agent = DQNagent.agent(state_size,action_size,gamma=0.999 , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.996, learning_rate=0.001, batch_size=128)

if rank == 0:
    w_model=agent.model.get_weights()
    comm.send(w_model, dest=1, tag=12)



elif rank == 1:
        weights = comm.recv(source=0, tag=12)
        agent.model.set_weights(weights)
