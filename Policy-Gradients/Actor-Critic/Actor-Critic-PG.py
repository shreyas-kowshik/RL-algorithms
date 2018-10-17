
# coding: utf-8

# In[287]:


import gym
import numpy as np
import tensorflow as tf
import time
import math

# In[311]:

'''
For the CartPole-v0 environment, actor critic is not able to generalise as such as the reward funcition is always +1
And the TD-error is used as the advantage function
Now the critic tires to bring down the difference between V(s) and V(s') to approximately the reward for gamma=0.95
It learns to keep this constant at 1.0 but then the policy does not see how good it is as there is no neg feedback
At each time step, the agent does not have enough information if it is doing the right thing
So it tends to learn any policy that can fetch it the next immediate reward and hence it can be wrong

Actor Critic solution tends to converge to a local maxima
Reiinforce finds the optimal solution

Experiment : Try out with TD(n) instead of TD(0) as the advantage function
'''

# Create our policy network
class PolicyNetwork:
    '''
    Given a state, it outputs the action probabilities
    '''
    def __init__(self,state_size,action_size,learning_rate=1e-5,name='PolicyNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        self.build_model()
    
    def build_model(self):
        # Define placeholders
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE):
            with tf.name_scope(self.name + '_placeholders'):
                self.advantage = tf.placeholder(shape=[None],name='advantage',dtype=tf.float32)
                self.actions_ = tf.placeholder(shape=[None,self.action_size],name='actions_',dtype=tf.float32)
                self.state = tf.placeholder(shape=[None,self.state_size],name='state',dtype=tf.float32)
            
            # Build the network
            # We just have a 3 full connected layers...That's it!
            self.fc_1 = tf.layers.dense(self.state,128,name='fc_1'\
                                        ,kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.fc_2 = tf.layers.dense(self.fc_1,256,name='fc_2'
                                       ,kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.output = tf.layers.dense(self.fc_2,action_size,name='output',activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # Action probability distributio
            # It's NOT SIGMOID you fool!!!!!!!!!!!
            self.actions = tf.nn.softmax(self.output)
            
            # Take this as a supervised learning problem, we have the labels as the actions
            # self.neg_log_probs = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.actions_)
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,labels=self.actions_)*self.advantage)
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
    def predict(self,sess,state):
        '''
        Predict the action probabilities
        
        state : the numpy array of the state passed
        '''
        action_probs = sess.run(self.actions,feed_dict={self.state:state})
        return action_probs
    
    def update(self,sess,state,actions,advantage):
        '''
        Make the optimizer update
        state : the state for which we make the update
        '''
        _,loss = sess.run([self.optimizer,self.loss],feed_dict={self.state:state,self.actions_:actions,self.advantage:advantage})
        return loss


# In[312]:


class ValueEstimatorNetwork:
    '''
    This computes the generalized advantage approximated by the TD error on the State-Value function
    TD-error = r + gamma*V(s_t+1) - V(s_t)
    '''
    def __init__(self,state_size,learning_rate=1e-5,gamma=0.99,name = 'AdvantageNetwork'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.name = name
        self.build_model()
        
    def build_model(self):
        with tf.variable_scope(self.name):
                # Define placeholders
                with tf.name_scope('Placeholders'):
                    self.state = tf.placeholder(shape=[None,self.state_size],name='state',dtype=tf.float32)
                    self.target = tf.placeholder(shape=[None],name='target',dtype=tf.float32)
                
                # Build the network
                self.fc_1 = tf.layers.dense(self.state,128,name='fc_1',\
                                           kernel_initializer=tf.zeros_initializer())
                self.output = tf.layers.dense(self.fc_1,1,name='output',activation=None,kernel_initializer=tf.zeros_initializer())
                
                # output is the value estimate
                self.loss = tf.reduce_mean(tf.square(self.output - self.target))
                
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
    def predict(self,sess,state):
        '''
        Predict the value estimate
        
        state : the numpy array of the state passed
        '''
        value_estimate = sess.run(self.output,feed_dict={self.state:state})
        return value_estimate
    
    def update(self,sess,state,target):
        '''
        Make the optimizer update
        state : the state for which we make the update
        Update made : V(s) = r + gamma*V(s')
        '''
        _,loss = sess.run([self.optimizer,self.loss],feed_dict={self.state:state,self.target:target})
        return loss

class ActorCriticNetwork:
    def __init__(self,env,state_size,action_size,actor_lr=1e-5,critic_lr=1e-5,                 gamma=0.99,num_episodes=1000,episode_length=1000,                 render=False,name='ACNetwork'):
        self.env = env
        self.env.seed(1)
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.render = render

        tf.reset_default_graph()

        self.actor_network = PolicyNetwork(self.state_size,self.action_size,learning_rate=self.actor_lr,name='ActorNetwork')
        self.critic_network = ValueEstimatorNetwork(self.state_size,learning_rate=self.critic_lr,name='CriticNetwork')
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        writer = tf.summary.FileWriter('tmp')
        writer.add_graph(self.sess.graph)
        tf.summary.scalar("Actor Loss",self.actor_network.loss)
        tf.summary.scalar("Critic Loss",self.critic_network.loss)

        write_op = tf.summary.merge_all()

        all_rewards = []
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            
            # Episode Statistics
            total_reward = 0.0 
            
            while True:
                # print(self.sess.graph)

                if self.render is True:
                    self.env.render()

                # Choose an action
                action_probs = self.actor_network.predict(self.sess,state.reshape(1,self.state_size))

                action_probs = np.array(action_probs)
    
                action_probs = action_probs.reshape(self.action_size)
                
                action = np.random.choice(np.arange(len(action_probs)),p=action_probs)

                actions_one_hot = np.zeros((1,self.action_size)).astype(np.float32)
                actions_one_hot[0,action] = 1.0

                # Take a step in the environment
                next_state,reward,done,_ = self.env.step(action)

                # Compute The TD-Error to be used as the Advantage
                V_s_next = np.array(self.critic_network.predict(self.sess,next_state.reshape(1,self.state_size)).reshape(1))
                V_s = np.array(self.critic_network.predict(self.sess,state.reshape(1,self.state_size)).reshape(1))
                
                V_target = reward + self.gamma*V_s_next
                TD_error = V_target - V_s

                # Update the critic network
                critic_loss = self.critic_network.update(self.sess,state.reshape(1,self.state_size),V_target)
                
                # Update the actor network
                actor_loss = self.actor_network.update(self.sess,state.reshape(1,self.state_size),actions_one_hot,TD_error)
                
                # Update the statistics 
                total_reward+=reward
                mean_reward = np.sum(all_rewards)/(1.0*(episode + 1))

                if episode >= num_episodes:
                    # time.sleep(0.1)
                    print('---')
                    print(action_probs)
                    print(action)
                    print('Reward : ' + str(reward))
                    print('V(s) : {}\nV(s_next) : {}\nV_target : {}\nTD_error : {}'.format(V_s,V_s_next,V_target,TD_error))
                    print('Critic Loss : {}\nActor Loss : {}'.format(critic_loss,actor_loss))


                if done:
                    all_rewards.append(total_reward)
                    break


                state = next_state

            # break
            print(action_probs)
            mean_reward = np.sum(all_rewards)/(1.0*(episode + 1))
            max_reward = np.amax(all_rewards)
            print('Episode : {}\nTotal Reward : {}\nMean Reward : {}\nMax So Far : {}'.format(episode,total_reward,mean_reward,max_reward))

env = gym.make('CartPole-v0')
env = env.unwrapped

# For the cartpole environment
state_size = 4
action_size = 2

actorCriticNet = ActorCriticNetwork(env,state_size,action_size,num_episodes=10000,gamma=0.95,actor_lr=1e-5,critic_lr=1e-6,render=True)