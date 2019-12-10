from mpi4py import MPI
import gym
import numpy as np
from collections import deque
import DQN_mpi as DQNagent
import tensorflow as tf
from collections import deque
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

simulators = size-1

simulations = 500
rep_interval = 25
repetitions = int(simulations / (rep_interval*simulators))
rep_each = int(simulations / simulators)
weights = None

if rank == 0:
    start_time = time.time()

FILE_NAME = "ann-weights.h5"
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
if rank == 0:
    agent = DQNagent.agent(state_size,action_size,gamma=0.999 , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.95, learning_rate=0.001, batch_size=128)

#first simulation to have training data
if not rank == 0 :
    agent = DQNagent.simulator(state_size,action_size , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.95, batch_size=128)
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


        agent.reduce_random()

    print("episode: {}/{}, score: {}, e: {:.2}, mean_score: {}"
        .format(e, rep_each, score, agent.epsilon,mean_score))

#First gather and save the data to start the iterative process
data = comm.gather(agent.memory, root=0)
if rank == 0:
    sent_weights = [[]]*(size-1)
    w_model=np.copy(agent.model.get_weights())
    for d in data[1:]:
        agent.memory += d
    data = [[]]*(size-1)
    d = [[]]*(size-1)
    for i in range(1,size):
        data[i-1] = comm.Irecv(d, source=i, tag=12)
        comm.Isend(w_model, dest=i, tag=11)

else:
    weights= np.empty(np.asarray(agent.model.get_weights()).shape)
    network = comm.Irecv(weights, source=0, tag=11)


for i in range(repetitions):
    if rank == 0:
        #load weights to send
        for i in range(1,size):
            #chek if received new data
            flag, d = data[i-1].Test()
            if not d == None:
                #if not received we cancel to ask again for data
                #data[i-1].Cancel()
            #else:
                print('holaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',i)
                sent_weights[i-1]=agent.model.get_weights()
                data[i-1] = comm.Irecv(source=i, tag=12)
                comm.Isend(sent_weights[i-1], dest=i, tag=11)
                agent.memory += d
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
            agent.reduce_random()

        print("episode: {}/{}, score: {}, e: {:.2}, mean_score: {}"
            .format(e+(i+1)*rep_interval, rep_each, score, agent.epsilon,mean_score))
        network.Wait()
        print(weights)
