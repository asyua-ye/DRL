import numpy as np
import torch



class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size=256):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def load_D4RL(self, dataset):
        num_observations = dataset['observations'].shape[0]
        self.state[:num_observations] = dataset['observations']
        self.action[:num_observations] = dataset['actions']
        self.next_state[:num_observations] = dataset['next_observations']
        self.reward[:num_observations] = dataset['rewards'].reshape(-1, 1)
        self.not_done[:num_observations] = (1. - dataset['terminals'].reshape(-1, 1))
        self.size = num_observations
        self.ptr = num_observations % self.max_size

    
    
class priorReplayBuffer(object):
    """
    也可以用线段树实现
    不过后面再说把！
    
    这里是直接索引，时间大头在计算概率，o(n)
    而线段树，构建完毕后，更新和查询都是o(logn)
    
    """
    
    def __init__(self, state_dim, action_dim, max_size=int(25e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.max = 1.0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.prior = np.zeros((max_size, 1))
        self.ind = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.prior[self.ptr] = self.max
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def compute_sampling_probabilities(self):
        total_priority = np.sum(self.prior[:self.size])
        if total_priority == 0:
            return np.ones(self.size) / self.size 
        return self.prior[:self.size].reshape(-1) / total_priority


    def sample(self, batch_size=256):
        probabilities = self.compute_sampling_probabilities()
        ind = np.random.choice(np.arange(self.size), size=batch_size, p=probabilities)
        self.ind = ind

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            )
        
    def update(self,values):
        self.prior[self.ind] = values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)
        
    def load_D4RL(self, dataset):
        num_observations = dataset['observations'].shape[0]
        self.state[:num_observations] = dataset['observations']
        self.action[:num_observations] = dataset['actions']
        self.next_state[:num_observations] = dataset['next_observations']
        self.reward[:num_observations] = dataset['rewards'].reshape(-1, 1)
        self.not_done[:num_observations] = (1. - dataset['terminals'].reshape(-1, 1))
        self.prior[:num_observations] = self.max
        
        self.size = num_observations
        self.ptr = num_observations % self.max_size
    
    
    
    
    
    
    