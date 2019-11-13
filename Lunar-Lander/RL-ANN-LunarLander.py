# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000

GANMA = 0.999    # discount rate
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.00
EPSILON_DECAY = 0.996
LEARNING_RATE = 0.001

#TRAIN = 1 trains to a file. TRAIN = 0 plays with the coefficients of the file.
TRAIN = 1;
FILE_NAME = "ann-weights.h5"

if __name__ == "__main__":
  env = gym.make('LunarLander-v2')
  state_size = 8
  action_size = env.action_space.n

  #Create the artificial neural network (ann)
  ann = Sequential()

  ann.add(Dense(64, input_dim = state_size, activation='relu'))
  ann.add(Dense(64, activation='relu'))
  ann.add(Dense(action_size, activation='linear'))
  ann.compile(loss='mse', optimizer=Adam(LEARNING_RATE), metrics=['mae'])

  if TRAIN == 1:
    epsilon = EPSILON
    for e in range(EPISODES):
      #Initial state and action
      recomp = 0
      state = env.reset()
      state = np.reshape(state[0:state_size], [1, state_size])
      epsilon *= EPSILON_DECAY;
      done = False
      time = 0
      #choose greedy action
      action = np.argmax(ann.predict(state))
      while not done:
        #env.render()
        # Take that action and see the next state and reward
        new_state, reward, done, _ = env.step(action)
        new_state = np.reshape(new_state[0:state_size], [1, state_size])
        if done:
          target = reward
          target_vec = ann.predict(state)
          target_vec[0][action] = target
          ann.fit(state, target_vec, epochs=1, verbose=0)
          break
        # Select an new action (epsilon-greedy)
        if np.random.random() < epsilon:
            new_action = random.randrange(action_size)
        else:
            new_action = np.argmax(ann.predict(new_state)[0])
        # Train the model if the episode finished
        #Save the target value
        target = reward + GANMA * np.amax(ann.predict(new_state)[0])#[new_action]
        #Create list with old values for the state
        target_vec = ann.predict(state)
        #Update the value that we want to  change with the target
        target_vec[0][action] = target
        ann.fit(state, target_vec, epochs=1, verbose=0)
        state = new_state
        action = new_action
        # Count the time
        recomp += reward

      #print episode results
      print("episode: {}/{}, score: {}, e: {:.2}"
            .format(e, EPISODES, recomp, epsilon))


    for e in range(EPISODES):
        # Initial state and action
        state = env.reset()
        state = np.reshape(state[0:state_size], [1, state_size])
        done = False
        time = 0
        recomp = 0
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
            recomp += reward
        # print episode results
        print("episode: {}/{}, score: {},".format(e, EPISODES, recomp ))