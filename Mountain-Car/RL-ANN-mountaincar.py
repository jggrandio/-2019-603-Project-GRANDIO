# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#con el que funcionaba eran 600
EPISODES = 500

# parametros para el n-step
N=60




wins=0
recomp=0
GANMA = 0.999  # discount rate
EPSILON = 0.01  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 1
LEARNING_RATE = 0.001

#TRAIN = 1 trains to a file. TRAIN = 0 plays with the coefficients of the file.
TRAIN = 1;
FILE_NAME = "ann-weights.h5"

if __name__ == "__main__":
  env = gym.make('MountainCar-v0')
  state_size = 2
  action_size = env.action_space.n

  #Create the artificial neural network (ann)
  ann = Sequential()

  ann.add(Dense(64, input_dim = state_size, activation='linear'))
  ann.add(Dense(64, activation='relu'))
  ann.add(Dense(64, activation='relu'))
  ann.add(Dense(action_size, activation='linear'))
  ann.compile(loss='mse', optimizer=Adam(LEARNING_RATE), metrics=['mae'])

  if TRAIN == 1:
    epsilon = EPSILON
    for e in range(EPISODES):
      #Initial state and action
      R = deque()
      A = deque()
      S = deque()
      tau = 0
      T = float('inf')
      t = 0
      recomp = 0
      state = env.reset()
      state = np.reshape(state[0:state_size], [1, state_size])
      new_state=state
      epsilon *= EPSILON_DECAY;
      done = False
      time = 0
      reward= -1
      #choose greedy action
      action = np.argmax(ann.predict(state))
      S.append(state)
      A.append(action)

      while tau<T:
        if recomp < -5000:
          env.render()
        if new_state[0][0]<0.5:
          new_state, reward, done, _ = env.step(action)
          new_state = np.reshape(new_state[0:state_size], [1, state_size])
          R.append(reward)
          S.append(new_state)
          if new_state[0][0]>=0.5:
            T=t+1
          else:
            if np.random.random() < epsilon:
              new_action = random.randrange(action_size)
            else:
              new_action = np.argmax(ann.predict(new_state)[0])


        tau= t-N+1
        # Select an new action (epsilon-greedy)
        # Train the model if the episode finished
        if tau > 0:
          G = 0.0
          for i in range(tau+1,min(tau+N,T)):
            G+= (GANMA**(i-tau-1))*R[i]
          if(tau + N < T):
            G = G + (GANMA ** N) *np.amax(ann.predict(S[tau+N])[0])

          target =GANMA * G#[new_action]
          target_vec = ann.predict(S[tau])
          target_vec[0][A[tau]] = target
          ann.fit(S[tau], target_vec, epochs=1, verbose=0)
        state = new_state
        action = new_action
        A.append(action)
        # Count the time
        recomp += reward
        t += 1
      if recomp>-199:
          wins+=1
      #print episode results
      print("episode: {}/{}, score: {}, e: {:.2}, wins : {}"
          .format(e, EPISODES, recomp, epsilon, wins))


    for e in range(EPISODES):
        # Initial state and action
        state = env.reset()
        state = np.reshape(state[0:4], [1, state_size])
        done = False
        time = 0
        # choose greedy action
        action = np.argmax(ann.predict(state))
        while not done:
            env.render()
            # Take that action and see the next state and reward
            new_state, reward, done, _ = env.step(action)
            new_state = np.reshape(new_state[0:state_size], [1, state_size])
            # Select an new action (epsilon-greedy)
            new_action = np.argmax(ann.predict(new_state)[0])
            # Train the model if the episode finished
            state = new_state
            action = new_action
            # Count the time
            time += 1
        # print episode results
        print("episode: {}/{}, score: {},".format(e, EPISODES, time ))