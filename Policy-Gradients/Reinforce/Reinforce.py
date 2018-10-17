import gym
import numpy as np
import tensorflow as tf
import time
import math

'''
For the CartPole-v0 environment, actor critic is not able to generalise as such as the reward funcition is always +1
And the TD-error is used as the advantage function
Now the critic tires to bring down the difference between V(s) and V(s') to approximately the reward for gamma=0.95
It learns to keep this constant at 1.0 but then the policy does not see how good it is as there is no neg feedback
At each time step, the agent does not have enough information if it is doing the right thing
So it tends to learn any policy that can fetch it the next immediate reward and hence it can be wrong

It works with MC Reinforce that uses the total return for all timesteps in the episode
ReLU + xavier - faster convergence
ReLU + zeros - really slow, vanishing gradient
ReLU - even on removing, fast convergence, it has little role to play - learnt around 150 episodes only!
'''

'''
Implemented as a modified version of actor critic
Here the advantage of the critic is the discounted reward for each time-step in the episode
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
            self.fc_1 = tf.contrib.layers.fully_connected(inputs=self.state,num_outputs=10,activation_fn=None\
                                        ,weights_initializer=tf.contrib.layers.xavier_initializer())
            self.fc_2 = tf.contrib.layers.fully_connected(inputs=self.fc_1,num_outputs=self.action_size,activation_fn=None\
                                        ,weights_initializer=tf.contrib.layers.xavier_initializer())
            self.output = tf.contrib.layers.fully_connected(inputs=self.fc_2,num_outputs=self.action_size,activation_fn=None\
                                        ,weights_initializer=tf.contrib.layers.xavier_initializer())
            
            # Action probability distributio
            # It's NOT SIGMOID you fool!!!!!!!!!!!
            self.actions = tf.nn.softmax(self.output)
            
            # Take this as a supervised learning problem, we have the labels as the actions
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.output, labels = self.actions_)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.advantage) 
            
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

class ReinforceNetwork:
    def __init__(self,env,state_size,action_size,actor_lr=1e-5,gamma=0.99,num_episodes=1000,episode_length=1000,render=False,name='ACNetwork'):
        self.env = env
        self.env.seed(1)
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.render = render

        tf.reset_default_graph()

        self.actor_network = PolicyNetwork(self.state_size,self.action_size,learning_rate=self.actor_lr,name='ActorNetwork')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('tmp')
        writer.add_graph(self.sess.graph)
        tf.summary.scalar("Actor Loss",self.actor_network.loss)

        write_op = tf.summary.merge_all()

        all_rewards = []
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            
            # Episode Statistics
            total_reward = 0.0 
            episode_rewards = []
            episode_states = []
            episode_actions = []

            while True:
                if self.render is True:
                    self.env.render()

                # Choose an action
                action_probs = self.actor_network.predict(self.sess,state.reshape([1,self.state_size]))

                action_probs = np.array(action_probs)
                action = np.random.choice(range(action_probs.shape[1]),p=action_probs.ravel())
 
                actions_one_hot = np.zeros(self.action_size).astype(np.float32)

                actions_one_hot[action] = 1.0
                actions_one_hot = actions_one_hot.reshape(1,actions_one_hot.shape[0])

                # Take a step in the environment
                next_state,reward,done,_ = self.env.step(action)

                episode_rewards.append(reward)
                episode_states.append(state)
                episode_actions.append(actions_one_hot)

                # Update the statistics 
                total_reward+=reward
                mean_reward = np.sum(all_rewards)/(1.0*(episode + 1))

                if done:
                    discounted_reward = 0.0
                    discounted_rewards = []

                    for r in reversed(episode_rewards):
                        discounted_reward = r + self.gamma*discounted_reward 
                        print('Reward : ' + str(discounted_reward)) 
                        discounted_rewards.append(discounted_reward)
                    
                    # Try with normalization
                    discounted_rewards = np.array(discounted_rewards)
                    mean = np.mean(discounted_rewards)
                    std = np.std(discounted_rewards)
                    discounted_rewards = (discounted_rewards - mean)/(1.0*std)

                    discounted_rewards = discounted_rewards[::-1]

                    episode_states = np.vstack(episode_states)
                    episode_actions = np.vstack(episode_actions)
                    print(episode_states.shape)
                    print(episode_actions.shape)

                    actor_loss = self.actor_network.update(self.sess,np.vstack(episode_states),np.vstack(episode_actions)\
                        ,discounted_rewards)
                    print('---')
                    print(action_probs)
                    print('Actor Loss : {}'.format(actor_loss))

                    all_rewards.append(total_reward)
                    break


                state = next_state

            # break
            mean_reward = np.sum(all_rewards)/(1.0*(episode + 1))
            max_reward = np.amax(all_rewards)
            print('Episode : {}\nTotal Reward : {}\nMean Reward : {}\nMax So Far : {}'.format(episode,total_reward,mean_reward,max_reward))

env = gym.make('CartPole-v0')
env = env.unwrapped

# For the cartpole environment
state_size = 4
action_size = 2

reinforceNet = ReinforceNetwork(env,state_size,action_size,num_episodes=10000,gamma=0.95,actor_lr=1e-2,render=True)


