#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:19:07 2021

@author: intern
"""

import gym
import numpy as np
import random
from IPython.display import clear_output


alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []
env = gym.make("Taxi-v3").env

env.render()

env.reset()
env.render()
q_table = np.zeros([env.observation_space.n, env.action_space.n])





print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))



frames = []
epochs = 0
penalties, reward = 0, 0 
done = False



state = env.encode(3, 1, 2, 0)
print("State:", state)
env.s = state
env.render()
env.P[328] ## action: [(probability, nextstate, reward, done)]
env.s = 328



""" Training the agent """

for i in range(1, 100001):
    state = env.reset()
    
    epochs, penalties, reward = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done, info = env.step(action)
        
        old_value = q_table[state, action] # save current value
        next_max = np.max(q_table[next_state]) # what's most beneficial if we take this step
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value # update the table
        
        if reward == -10:
            penalties += 1
            
        state = next_state
        epochs += 1
        
    if i % 1000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
    
print("Training finished.\n")





total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, rweward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        
        if reward == -10:
            penalties += 1
            
        epochs += 1
        
    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Penalties per episode: {total_penalties / episodes}")














#while not done:
#    action = env.action_space.sample()
#    state, reward, done, info = env.step(action)
#    
#    if reward == -10:
#        penalties += 1
#        
#    # Put each rendered frame into dict for animation
#    frames.append({
#            'frame': env.render(mode='ansi'),
#            'state': state,
#            'action': action,
#            'reward': reward
#            }
#    )
#    
#    epochs += 1
#    
#print("Timesteps taken: {}".format(epochs))
#print("Penalties incurred: {}".format(penalties))





#from time import sleep
#
#def print_frames(frames):
#    for i, frame in enumerate(frames):
#        clear_output(wait=True)
#        print(frame['frame'].getvalue())
#        print(f"Timestep: {i + 1}")
#        print(f"State: {frame['state']}")
#        print(f"Action: {frame['action']}")
#        print(f"Reward: {frame['reward']}")
#        sleep(.1)
#
#print_frames(frames)
