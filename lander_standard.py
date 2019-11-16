# -*- coding: utf-8 -*-
import gym
import numpy as np
from collections import deque
import DQNagent

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

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNagent.agent(state_size,action_size,gamma=0.999 , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.992, learning_rate=0.001, batch_size=128)
    scores = deque(maxlen=100)
    mean_score = 0
    for e in range(EPISODES):
      #Initial state
      state = env.reset()
      state = agent.format_state(state)
      done = False
      score = 0

      while not done:
        #if mean_score < -500:
            #env.render()
        # env.render()
        # Take that action and see the next state and reward
        action = agent.action(state)
        new_state, reward, done, _ = env.step(action)
        new_state = agent.format_state(new_state)
        agent.remember(state, action, reward, new_state, done)
        agent.training(state, action, reward, new_state, done)
        state= new_state
        score += reward
      scores.append(score)
      mean_score = np.mean (scores)
      #print episode results

      print("episode: {}/{}, score: {}, e: {:.2}, mean_score: {}"
            .format(e, EPISODES, score, agent.epsilon,mean_score))

      if mean_score >=195 and e>=100:
          print('Solved after {} episodes'.format(e))
          break
      agent.reduce_random()
      agent.replay()
      agent.soft_update_target_network()
