import gym
import numpy as np
import tensorflow as tf
import cv2
import random

from PER import PER
from StateProcessor import StateProcessor
from DQNAgent import DQNAgent

state_size = [84,80,4]
action_size = 4
num_steps = 10000 # Run for these many steps

processor = StateProcessor()

tf.reset_default_graph()
sess = tf.Session()

agentNetwork = DQNAgent(state_size,action_size,'Agent')
targetNetwork = DQNAgent(state_size,action_size,'Target')

# Load the model
def load_model(sess):
    saver = tf.train.Saver()
    saver.restore(sess,'tmp/breakout/model/model.ckpt')

load_model(sess)

processor.reset()
for i in range(num_steps):
    state = processor.get_state()
    processor.render()

    Q = sess.run([agentNetwork.Q_all_a],feed_dict={agentNetwork.state:state.reshape((1,*state_size))})            
    action = np.argmax(Q)
    print(action)
    next_state,reward,done = processor.step(action)

    if done:
        processor.reset()
