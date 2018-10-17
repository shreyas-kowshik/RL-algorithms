import numpy as np 
import tensorflow as tf 

'''
Implementation of a Deuling Network Architecture for a DQN Agent
'''

class DQNAgent:
    def __init__(self,state_size,action_size,memory_size=100000,learning_rate=1e-5,name='DQNAgent'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.name = name
        
        self.build_model()
    
    def build_model(self):
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            '''
            Our state consists of four stacked frames of the game of shape - (WIDTH,HEIGHT,4)
            '''
            # For the Importance Sampling Weights
            self.weights_placeholder = tf.placeholder(shape=[None,1],name='weights_placeholder',dtype=tf.float32)
            self.state = tf.placeholder(shape=[None,*self.state_size],name='state',dtype=tf.float32)
            
            # actions_ is a one hot vector
            self.actions_ = tf.placeholder(shape=[None,self.action_size],name='actions_',dtype=tf.float32) 
            
            # Our target Q value
            self.target_Q = tf.placeholder(shape=[None],name='target_Q',dtype=tf.float32)

            self.conv1 = tf.nn.elu(tf.layers.conv2d(self.state,filters=32,kernel_size=[4,4],strides=[4,4],name='conv1'),name='elu1')
            self.conv2 = tf.nn.elu(tf.layers.conv2d(self.conv1,filters=64,kernel_size=[4,4],strides=[2,2],name='conv2'),name='elu2')
            self.conv3 = tf.nn.elu(tf.layers.conv2d(self.conv2,filters=128,kernel_size=[2,2],strides=[2,2],name='conv3'),name='elu3')
            self.flat = tf.layers.flatten(self.conv3)
            
            # Branch out now to V(s) and A(s,a)
            self.V_fc = tf.layers.dense(self.flat,512,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='V_fc')
            self.V = tf.layers.dense(self.V_fc,1,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='V')
            
            self.A_fc = tf.layers.dense(self.flat,512,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='A_fc')
            self.A = tf.layers.dense(self.V_fc,self.action_size,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='A')
            
            # This stores the Q values for all possible actions
            self.Q_all_a = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keepdims=True))
            
            # This keeps the values only for the action taken - Q[s,a] in the transition tuple
            self.Q = tf.reduce_sum(tf.multiply(self.Q_all_a,self.actions_),axis=1)
            
            self.TD_error = tf.abs(self.target_Q - self.Q)

            self.loss = tf.reduce_mean(self.weights_placeholder*tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)