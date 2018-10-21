
# coding: utf-8

# In[1]:


import gym
import numpy as np
import tensorflow as tf


# In[3]:


env = gym.make('Breakout-v0')

num_steps = 10
nA = 4 # Number of actions

state = env.reset()
print(env.unwrapped.get_action_meanings())
print(state.shape)
print(state)
# for _ in range(num_steps):
#     env.render()
#     state,_,done,_ = env.step(2) # take a random action
#     print(state.shape)
#     if(done):
#     	env.reset()

# print('Done with basic render check...')
# del env