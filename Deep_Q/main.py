import gym
import numpy as np
import tensorflow as tf
import cv2
import random

from PER import PER
from StateProcessor import StateProcessor
from DQNAgent import DQNAgent

'''
Some Notable Issue During Training : 
The block remains at one end only for a long time.This is because a random spawn of the ball hits it and it gets good rewards following that.
It is not exploring the possibility of moving and scoring more.This is sorts of a local optima.
'''
# Main function
# Let's create all the stuff now

# Constants
state_size = [84,80,4]
action_size = 4
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9998
mem_size = 100000
batch_size = 64
num_episodes = 50000
episode_length = 5000 # End an episode after these steps
network_copy_frequency = 10000
network_copy_frequency_step = 0
pretrain_length = 100000

# Variables
processor = StateProcessor()
memory = PER(mem_size)

# Variables to save
reward_holder = []

tf.reset_default_graph()
sess = tf.Session()

agentNetwork = DQNAgent(state_size,action_size,'Agent')
targetNetwork = DQNAgent(state_size,action_size,'Target')

# # Tensorboard visualiser
# writer = tf.summary.FileWriter('/tmp/breakout/1')
# writer.add_graph(sess.graph)
writer = tf.summary.FileWriter("/tmp/breakout/1")

## Losses
tf.summary.scalar("Loss", agentNetwork.loss)
write_op = tf.summary.merge_all()

def choose_action(state,epsilon,epsilon_decay):
    '''
    Choose epsilon greedy action
    '''
    rand = np.random.rand()
    
    if rand < epsilon:
        # Select a random action
        action = random.randint(0,action_size - 1)
        # print('Action : ' + str(action))
        # print('Epsilon : ' + str(epsilon))
        epsilon*=epsilon_decay

    # print('Epsilon : ' + str(epsilon))

    else:
        # Select a = argmax_a(Q[s,a])
        Q = sess.run([agentNetwork.Q_all_a],feed_dict={agentNetwork.state:state.reshape((1,*state_size))})
        action = np.argmax(Q)
    
    if epsilon < 0.1:
        epsilon = 0.1

    return action,epsilon

print('Filling Buffer Memory')
# Fill our memory with dummy values
for i in range(pretrain_length):
        # print(i)
        state = processor.get_state()
        action = processor.sample_env_action()
        next_state,reward,done = processor.step(action) # take a random action
        if done:
            processor.reset()
        memory.store((state,action,reward,next_state,done))  
print('Filled Buffer Memory')

# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():
    
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Agent')
    
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Target')
    
    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def save_model(sess):
	saver = tf.train.Saver()
	saver.save(sess,'tmp/breakout/model/model.ckpt')

sess.run(tf.global_variables_initializer())
# print('Done')
sess.run(tf.local_variables_initializer())

saver = tf.train.Saver()

for episode in range(num_episodes):
    state = processor.reset()
    
    total_rewards = 0.0
    
    # Collect experience
    for _ in range(episode_length):
        state = processor.get_state()
        processor.render()

        network_copy_frequency_step+=1
        
        action,epsilon = choose_action(state,epsilon,epsilon_decay)
        
        next_state,reward,done = processor.step(action)
        # print('Step Done')
        if done:
            memory.store((state,action,reward,np.zeros((state_size[0],state_size[1],state_size[2]))\
                          ,done)) # No next state is there 
            total_rewards+=reward
            break
        
        total_rewards+=reward
        
        memory.store((state,action,reward,next_state,done)) 
    
    # Train our agent
    
    # Sample a minibatch of transition tuples
    batch_idx,min_batch,w = memory.sample_minibatch(batch_size)
    
    # Seperate out our mini-batch variables
    states = np.array([min_batch[i][0] for i in range(len(min_batch))])
    actions = np.array([min_batch[i][1] for i in range(len(min_batch))])
    rewards = np.array([min_batch[i][2] for i in range(len(min_batch))]).reshape(-1)
    next_states = np.array([min_batch[i][3] for i in range(len(min_batch))])
    dones = np.array([min_batch[i][4] for i in range(len(min_batch))])
    
    Qs_next_state = sess.run([agentNetwork.Q_all_a]\
                             ,feed_dict={agentNetwork.state:next_states})
    
    # Convert actions in current state of transition tuple to a one hot vector
    actions_one_hot_batch = np.zeros((batch_size,action_size))
    for i in range(batch_size):
        actions_one_hot_batch[i,actions[i]] = 1
    
    # # Convert actions in next state to a one hot vector
    # actions_ns_one_hot_batch = np.zeros((batch_size,action_size))
    # for i in range(batch_size):
    #     actions_ns_one_hot_batch[i,actions_next_state[i]] = 1
        
    target_Q_next_state = sess.run([targetNetwork.Q_all_a],feed_dict={targetNetwork.state:next_states})

    target_Q = np.zeros(batch_size)
    
    for i in range(batch_size):
        action = np.argmax(Qs_next_state[0][i])

        if dones[i] is True:
            target_Q[i] = rewards[i]
        else:
            target_Q[i] = rewards[i] + gamma*target_Q_next_state[0][i][action]
    
    # Make optimizer updates
    _,loss,TD_errors = sess.run([agentNetwork.optimizer,agentNetwork.loss,agentNetwork.TD_error],\
                     feed_dict={agentNetwork.state:states,agentNetwork.actions_:actions_one_hot_batch\
                               ,agentNetwork.target_Q:target_Q,agentNetwork.weights_placeholder:w})
    
    # Update the priorites in the memory
    memory.update(batch_idx,TD_errors)
    
    # Write TF Summaries
    summary = sess.run(write_op, feed_dict={agentNetwork.state: states,
                                       agentNetwork.target_Q: target_Q,
                                       agentNetwork.actions_: actions_one_hot_batch,
                                  agentNetwork.weights_placeholder: w})
    writer.add_summary(summary, episode)
    writer.flush()

    print('Episode : ' + str(episode) + '\nTotal Rewards : ' + str(total_rewards) + '\nLoss : ' + str(loss))
    print('TD Errors : ' + str(sum(TD_errors)))
    
    if episode%10 == 0:
    	save_model(sess)

    if network_copy_frequency_step > network_copy_frequency:
        network_copy_frequency_step = 0
        update_op = update_target_graph()
        sess.run(update_op)
        print('Model Updated')
