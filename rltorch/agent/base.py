import torch
from torch import nn
from copy import deepcopy


class BaseAgent:

    def __init__(self):
        self.env = None
        self.device = None
        self.shared_memory = None
        self.shared_weights = None
        self.net = nn.Sequential()
        self.target_net = nn.Sequential()
        self.memory = None

    def run(self):
        raise Exception('You need to implement run method.')

    def act(self, state):
        raise Exception('You need to implement act method.')

    def explore(self, state):
        raise Exception('You need to implement explore method.')

    def exploit(self, state):
        raise Exception('You need to implement explore method.')

    def interval(self):
        raise Exception('You need to implement interval method.')

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        raise Exception('You need to implement calc_current_q method.')

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        raise Exception('You need to implement calc_current_q method.')

    def load_weights(self):
        self.net.load_state_dict(self.shared_weights['net'])
        self.target_net.load_state_dict(self.shared_weights['target_net'])

    def save_weights(self):
        self.shared_weights['net'] = deepcopy(self.net).cpu().state_dict()
        self.shared_weights['target_net'] =\
            deepcopy(self.target_net).cpu().state_dict()

    def load_memory(self):
        while not self.shared_memory.empty():
            batch = self.shared_memory.get()
            self.memory.load_memory(batch)

    def save_memory(self):
        self.shared_memory.put(self.memory.get())
        self.memory.reset()

    def to_batch(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor([action]).unsqueeze(-1).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(-1).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).unsqueeze(-1).to(self.device)
        return state, action, reward, next_state, done
