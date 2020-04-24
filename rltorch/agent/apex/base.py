from copy import deepcopy
import numpy as np
import torch
from torch import nn

from rltorch.agent import BaseAgent


class ApexAgent(BaseAgent):

    def __init__(self):
        super(ApexAgent, self).__init__()
        self.net = nn.Sequential()
        self.target_net = nn.Sequential()
        self.epsilon = None
        self.writer = None
        self.gamma_n = None

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.explore(state)
        else:
            return self.exploit(state)

    def explore(self, state):
        return self.env.action_space.sample()

    def exploit(self, state):
        state = \
            torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.0
        with torch.no_grad():
            action = self.net(state).argmax().item()
        return action

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q = self.net(states).gather(1, actions.long()).view(-1, 1)
        return curr_q

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_action = torch.argmax(self.net(next_states), 1).view(-1, 1)
            next_q = self.target_net(
                next_states).gather(1, next_action.long()).view(-1, 1)

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q
        return target_q

    def load_weights(self):
        try:
            self.net.load_state_dict(self.shared_weights['net'])
            self.target_net.load_state_dict(self.shared_weights['target_net'])
            return True
        except KeyError:
            return False

    def save_weights(self):
        self.shared_weights['net'] = deepcopy(self.net).cpu().state_dict()
        self.shared_weights['target_net'] =\
            deepcopy(self.target_net).cpu().state_dict()

    def __del__(self):
        self.writer.close()
        self.env.close()
