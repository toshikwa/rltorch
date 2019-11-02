import numpy as np
import torch

from ..base import BaseAgent


class ApexAgent(BaseAgent):

    def __init__(self):
        super(ApexAgent, BaseAgent).__init__()
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
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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

    def __del__(self):
        self.writer.close()
        self.env.close()
