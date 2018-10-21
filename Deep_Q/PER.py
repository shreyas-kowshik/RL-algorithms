import gym
import numpy as np
import tensorflow as tf

'''
Prioritized Experience Replay

TODO : Add docstrings
'''

class SumTree:
    # Visualised in the form a binray heap
    
    def __init__(self,size):
        self.leaves = size # Number of leaf nodes in the sum tree
        self.size = 2*size - 1 # Number of nodes in the sum tree
        self.data_pointer = 0
        self.tree = np.ones((self.size))
        self.data = np.zeros((self.size),dtype=object) # It stores an experience tuple
        
    def add(self,priority,data):
        tree_index = self.data_pointer + self.leaves - 1 # Refers to the leaf node number indexed by data_pointer

        self.data[self.data_pointer] = data
        
        self.update(tree_index,priority)
        
        #Updating to the next index
        self.data_pointer+=1
        
        # Checking for overflow
        if self.data_pointer >= self.leaves:
            self.data_pointer = 0
    
    def update(self,tree_index,priority):
        # print('In Update---')
        # print('Tree Index : ' + str(tree_index))
        # print('Priority : ' + str(priority))
        self.tree[tree_index] = priority
        
        # Update the non-leaf nodes up the tree
        parent = ((tree_index + 1)//2) - 1

        while parent >= 0:
            parent = ((tree_index + 1)//2) - 1
            
            if parent < 0:
                break

            left_child = 2*parent + 1
            right_child = left_child + 1
            self.tree[parent] = self.tree[left_child] + self.tree[right_child]
            tree_index = parent

    def get_leaf(self,s):
        # Return the leaf node such that the cumulative priority till that is <= s
        # Values are returned as index,data,priority
        # In short this 
        index = 0 # Start from the root
        leaf_index = None
        
        while True:
            left_child = 2*index + 1
            right_child = left_child + 1
            
            if left_child >= self.size: # index is the leaf node
                leaf_index = index
                break
            else:
                if self.tree[left_child] >= s:
                    index = left_child
                else:
                    s = s - self.tree[left_child]
                    index = right_child
        
        return leaf_index,self.data[leaf_index - self.leaves + 1],self.tree[leaf_index]
    
    def get_total_priority(self):
        return self.tree[0] # The root value

class PER:
    def __init__(self,size):
        self.size = size
        self.tree = SumTree(size)
        
        # PER constants
        self.alpha = 0.6
        self.beta = 0.6
        self.beta_increment_factor = 0.001
        self.epsilon = 0.01
        
        self.maximum_absolute_error = 1.0 # To clip the TD errors
        
    def store(self,experience):
        # experience : a tuple of (s,a,r,s')
        max_priority = np.max(self.tree.tree[-self.tree.leaves:])

        if max_priority == 0:
            max_priority = 1.0 # Add any random value for now as it will in due course be updated
        
        # We use maximum  priority for a new experience to ensure that it will be use at least once
        # This breaks the bias
        self.tree.add(max_priority,experience)
    
    def sample_minibatch(self,k):
        '''
        Parameters:
        k : mini-batch size
        
        Returns:
        batch_idx : index values of the SumTree of the nodes which were sampled so that their errors can be updated
        mini_batch : array of experience tuples in the mini-batch
        w : importance sampling weights
        '''
    
        # print('PER beta : ' + str(self.beta))
        # Samples a minibatch of size k
        mini_batch = []
        batch_idx = np.zeros((k))
        
        total_priority = self.tree.get_total_priority()
        
        # Divide the total probability in bins
        num_bins = total_priority/k
        
        # Importance Sampling Weights
        w = np.zeros((k,1))
        
        P_min = np.min(self.tree.tree[-self.tree.leaves:])/self.tree.get_total_priority()
        # print('P_min : ' + str(P_min))
        max_w = (1.*k*P_min)**(-self.beta)
        # print('Max w : ' + str(max_w))
        
        for i in range(k):
            # Find the bin priority indices
            left_index = i*num_bins
            right_index = (i+1)*num_bins
            s = np.random.uniform(left_index,right_index)
            
            # Sample a random node with priority in range [i*num_bins,(i+1)*num_bins]
            # This samples a node from a probability distribution represented by the priorities
            idx,data,priority = self.tree.get_leaf(s)
            
            mini_batch.append(data)
            
            # Calculate P(i)
            # We assume that the values in the tree are already exponentiated while updation
            # i.e the tree leaves store (p(i))**alpha
            sampling_probability = priority/self.tree.get_total_priority()
            
            # Calculate the weights
            # print('Priority : ' + str(priority))
            # print('Sampling Probability : ' + str(sampling_probability))
            # print('Calculated weight : ' + str(np.power(1.*k*sampling_probability,-self.beta)/(max_w)))
            w[i,0] = (1.*k*sampling_probability)**(-self.beta)/(max_w)
            
            # Set the index of the tree leaf
            batch_idx[i] = idx
        
        # Increment beta at each sampling step
        self.beta = np.min([1.,self.beta + self.beta_increment_factor])
        
        return batch_idx.astype(np.int),mini_batch,w
    
    def update(self,batch_idx,errors):
        '''
        errors : TD errors
        Updates the indices which were sampled with their corrected error values
        '''
        
        absolute_errors = np.abs(errors) + self.epsilon
        
        # Store (p(i))**alpha in the tree leaves
        clipped_errors = np.minimum(absolute_errors,self.maximum_absolute_error) # Clipped in range [0,1]
        exponentiated_errors = np.power(clipped_errors,self.alpha)
        
        # print('In Loop')
        for i,delta in zip(batch_idx,exponentiated_errors):
            # print(i)
            # print(delta)
            self.tree.update(i,delta)
    
    def print_priorities(self):
        print('Number of experiences stored : ' + str(len(self.tree.tree[-self.tree.leaves:])))
        print('Priorities : ' + str(self.tree.tree[-self.tree.leaves:]))