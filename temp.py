
from mpi4py import MPI
import gym
import numpy as np
from collections import deque
import DQNagent
import tensorflow as tf
from collections import deque

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

simulators = size-1

simulations = 5000
rep_interval = 25
repetitions = int(simulations / (rep_interval*simulators))
rep_each = int(simulations / simulators)

FILE_NAME = "ann-weights.h5"
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNagent.agent(state_size,action_size,gamma=0.999 , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.992, learning_rate=0.001, batch_size=128)

if rank == 0:
    data = agent.model.get_weights()
else:
    data = None
data = comm.bcast(data, root=0)



















'''from mpi4py import MPI
import gym
import numpy as np
from collections import deque
import DQNagent
import tensorflow as tf
from collections import deque

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

simulators = size-1

simulations = 5000
rep_interval = 25
repetitions = int(simulations / (rep_interval*simulators))
rep_each = int(simulations / simulators)

FILE_NAME = "ann-weights.h5"
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNagent.agent(state_size,action_size,gamma=0.999 , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.992, learning_rate=0.001, batch_size=128)

#first simulation to have training data
if not rank == 0 :
    for e in range(rep_interval):
        state = env.reset()
        state = agent.format_state(state)
        done = False
        score = 0
        while not done:
            scores = deque(maxlen=100)
            mean_score = 0
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
            .format(e, rep_each, score, agent.epsilon,mean_score))
        agent.reduce_random()

    print('simulation done')

data = comm.gather(agent.memory, root=0)

for i in range(repetitions):
    if rank == 0:
        for d in data[1:]:
            agent.memory += d
            print('len d:',len(d))
        for e in range(rep_interval*simulators):
            agent.replay()
            agent.soft_update_target_network()
        print('neuron trained')
    else:
        agent.memory=deque()
        for e in range(rep_interval):
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
                .format(e+(i+1)*rep_interval, rep_each, score, agent.epsilon,mean_score))
            agent.reduce_random()
        print('simulation done')
    data = comm.gather(agent.memory, root=0)

'''

'''
for int i in range(repetitions)
if rank == 0:
    data = comm.recv(source=1, tag=11)
    agent.memory += data
    w_model=agent.model.get_weights()
    comm.send(w_model, dest=1, tag=12)
    for e in range(EPISODES):
        agent.replay()
        agent.soft_update_target_network()
        if e % 25 == 0:
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






from mpi4py import MPI
import numpy as np
import time
import DQNagent


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


agent = DQNagent.agent(5,5,gamma=0.999 , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.992, learning_rate=0.001, batch_size=128)


if rank == 0:
    data = agent.model.get_weights()
else:
    data = None

data = comm.bcast(data, root=0)
if rank == 1:
    print(data)

'''
'''
if rank == 0:
    print(buffer[0])
    fh.Write(buffer)
if rank == 1:

    time.sleep(1)
    fh.Read (buffer)
    print(buffer)

fh.Close()
'''
