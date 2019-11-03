from copy import deepcopy
import torch
from torch import nn

from rltorch.agent import BaseAgent


class SacDiscreteAgent(BaseAgent):

    def __init__(self):
        super(SacDiscreteAgent, self).__init__()
        self.writer = None
        self.gamma_n = None
        self.alpha = None
        self.tau = None
        self.start_steps = None
        self.steps = None
        self.policy = nn.Sequential()
        self.critic = nn.Sequential()
        self.critic_target = nn.Sequential()

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, action = self.policy.sample(state)
        return action.item()

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())

        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_action_probs, log_next_action_probs, _ =\
                self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_q = next_action_probs * (
                next_q - self.alpha * log_next_action_probs)
            next_q = next_q.mean(dim=1).unsqueeze(-1)

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def soft_update(self):
        for target, source in zip(
                self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(
                target.data * (1.0 - self.tau) + source.data * self.tau)

    def hard_update(self):
        self.critic.load_state_dict(self.critic_target.state_dict())

    def load_weights(self):
        try:
            self.policy.load_state_dict(self.shared_weights['policy'])
            self.critic.load_state_dict(self.shared_weights['critic'])
            self.critic_target.load_state_dict(
                self.shared_weights['critic_target'])
            self.alpha = torch.tensor(
                self.shared_weights['alpha'], device=self.device)
            return True
        except KeyError:
            return False

    def save_weights(self):
        self.shared_weights['policy'] = deepcopy(
            self.policy).cpu().state_dict()
        self.shared_weights['critic'] = deepcopy(
            self.critic).cpu().state_dict()
        self.shared_weights['critic_target'] = deepcopy(
            self.critic_target).cpu().state_dict()
        self.shared_weights['alpha'] = self.alpha.clone().detach().item()

    def to_batch(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor([action]).unsqueeze(-1).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(-1).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).unsqueeze(-1).to(self.device)
        return state, action, reward, next_state, done

    def __del__(self):
        self.writer.close()
        self.env.close()
