from mpi4py import MPI
import gym
import numpy as np
from collections import deque
import DQNagent
import tensorflow as tf
import timeit


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

EPISODES = 5000

GANMA = 0.99    # discount rate
EPSILON = 1.0  # exploration rate
#Works with min 0.01
EPSILON_MIN = 0.00
EPSILON_DECAY = 0.995
# works with learning rate 0.001
LEARNING_RATE = 0.001

#TRAIN = 1 trains to a file. TRAIN = 0 plays with the coefficients of the file.
TRAIN = 1;
FILE_NAME = "ann-weights.h5"

env = gym.make('Acrobot-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNagent.agent(state_size,action_size,gamma=0.999 , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.999, learning_rate=0.001, batch_size=128)

if rank == 0:
    data = comm.recv(source=1, tag=11)
    agent.memory += data
    w_model=agent.model.get_weights()
    comm.send(w_model, dest=1, tag=12)
    for e in range(2*EPISODES):
        agent.replay()
        agent.soft_update_target_network()
        if e % 50 == 0:
            print('neuron trained')
            data = comm.recv(source=1, tag=11)
            agent.memory += data
            w_model=agent.model.get_weights()
            comm.send(w_model, dest=1, tag=12)



elif rank == 1:
    scores = deque(maxlen=100)
    mean_score = 0
    for e in range(EPISODES):
        state = env.reset()
        state = agent.format_state(state)
        done = False
        score = 0
        while not done:
            action = agent.action(state)
            new_state, reward, done, _ = env.step(action)
            new_state = agent.format_state(new_state)
            agent.remember(state, action, reward, new_state, done)
            state= new_state
            score += reward
        scores.append(score)
        mean_score = np.mean (scores)
        #print episode results

        print("episode: {}/{}, score: {}, e: {:.2}, mean_score: {}"
            .format(e, EPISODES, score, agent.epsilon,mean_score))
        agent.reduce_random()
        if e % 25 == 0:
            print('simulation done')
            comm.send(agent.memory, dest=0, tag=11)
            weights = comm.recv(source=0, tag=12)
            agent.model.set_weights(weights)
