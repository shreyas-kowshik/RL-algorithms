import gym
import numpy as np
from collections import deque

class StateProcessor(object):
    '''
    State Processor for the Breakout-v0 environment
    '''
    def __init__(self,name='Breakout-v0'):
        self.name = name
        self.env = gym.make(name)
        self.stack_size = 4
        self.reset()
    
    def reset(self):
        self.is_new_episode = True
        self.state = self.env.reset()        
        self.state_stack  =  deque([np.zeros((84,84), dtype=np.int) for i in range(self.stack_size)], maxlen=4)
        self.stack_states()
    
    def stack_states(self):
        '''
        Stack the 4 previous frames
        '''
        
        if self.is_new_episode is True:
            self.state_stack.append(self.process(self.state))
            self.state_stack.append(self.process(self.state))
            self.state_stack.append(self.process(self.state))
            self.state_stack.append(self.process(self.state))
            
            self.is_new_episode = False
        else:
            self.state_stack.append(self.process(self.state))           
        
        self.stacked_frames = np.stack(self.state_stack,axis=2)
            
    def process(self,state):
        '''
        Takes as input a (210,160,3) image
        Processes it to (84,80,1) image
        '''
        image = np.array(state,dtype=np.uint8)
        return np.mean(image[::2,::2],axis=2).astype(np.uint8)[14:98,:]
        
    def step(self,action):
        self.state,reward,done,_ = self.env.step(action)
        self.stack_states()
        return self.stacked_frames,reward,done
    
    def get_state(self):
        return self.stacked_frames
    
    def sample_env_action(self):
        return self.env.action_space.sample()
    
    def render(self):
        self.env.render()