import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from .base import SacDiscreteAgent
from ...memory import Memory, MultiStepMemory, PrioritizedMemory
from ...policy import ConvCategoricalPolicy
from ...q_function import TwinedDiscreteConvQNetwork


class SacDiscreteLearner(SacDiscreteAgent):

    def __init__(self, env, log_dir, shared_memory, shared_weights,
                 batch_size=64, lr=0.0003, memory_size=1e5, gamma=0.99,
                 tau=0.005, multi_step=3, per=True, alpha=0.6, beta=0.4,
                 beta_annealing=0.001, clip_grad=5.0, num_epochs=4,
                 start_steps=20000, log_interval=1, memory_load_interval=5,
                 target_update_interval=1, model_save_interval=5,
                 eval_interval=1000, cuda=True, seed=0):
        self.env = env
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.shared_memory = shared_memory
        self.shared_weights = shared_weights

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.policy = ConvCategoricalPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.critic = TwinedDiscreteConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device)
        self.critic_target = TwinedDiscreteConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device).eval()

        self.hard_update()
        self.q1_optim = optim.Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.critic.Q2.parameters(), lr=lr)

        self.target_entropy = -np.log(1.0/self.env.action_space.n) * 0.98
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.save_weights()
        if per:
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                (1,), self.device, gamma, multi_step,
                is_image=False, alpha=alpha, beta=beta,
                beta_annealing=beta_annealing)
        elif multi_step == 1:
            self.memory = Memory(
                memory_size, self.env.observation_space.shape,
                (1,), self.device, is_image=False)
        else:
            self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                (1,), self.device, gamma, multi_step,
                is_image=False)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary', 'leaner')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.steps = 0
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.clip_grad = clip_grad
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.memory_load_interval = memory_load_interval
        self.model_save_interval = model_save_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

    def run(self):
        while len(self.memory) <= self.start_steps:
            self.load_memory()

        while True:
            self.steps += 1
            self.learn()
            self.interval()

    def learn(self):
        total_q1_loss = 0.
        total_q2_loss = 0.
        total_policy_loss = 0.
        total_entropy_loss = 0.
        total_mean_q1 = 0.
        total_mean_q2 = 0.
        total_mean_entropy = 0.

        for epoch in range(self.num_epochs):
            if self.per:
                batch, indices, weights = \
                    self.memory.sample(self.batch_size)
            else:
                batch = self.memory.sample(self.batch_size)
                weights = 1.

            q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
                self.calc_critic_loss(batch, weights)
            policy_loss, entropy = self.calc_policy_loss(batch, weights)
            entropy_loss = self.calc_entropy_loss(entropy, weights)

            self.update_params(
                self.q1_optim, self.critic.Q1.modules(), q1_loss,
                self.clip_grad)
            self.update_params(
                self.q2_optim, self.critic.Q2.modules(), q2_loss,
                self.clip_grad)
            self.update_params(
                self.policy_optim, self.policy.modules(), policy_loss,
                self.clip_grad)
            self.update_params(
                self.alpha_optim, None, entropy_loss)

            if self.per:
                self.memory.update_priority(indices, errors.cpu().numpy())

            self.alpha = self.log_alpha.exp()
            total_entropy_loss += entropy_loss.detach().item()
            total_q1_loss += q1_loss.detach().item()
            total_q2_loss += q2_loss.detach().item()
            total_policy_loss += policy_loss.detach().item()
            total_mean_q1 += mean_q1
            total_mean_q2 += mean_q2
            total_mean_entropy += entropy.detach().mean().item()

        self.writer.add_scalar(
            'loss/Q1', total_q1_loss/self.num_epochs, self.steps)
        self.writer.add_scalar(
            'loss/Q2', total_q2_loss/self.num_epochs, self.steps)
        self.writer.add_scalar(
            'loss/policy', total_policy_loss/self.num_epochs, self.steps)
        self.writer.add_scalar(
            'loss/alpha', total_entropy_loss/self.num_epochs, self.steps)
        self.writer.add_scalar(
            'stats/alpha', self.alpha.clone().item(), self.steps)
        self.writer.add_scalar(
            'stats/mean_Q1', total_mean_q1/self.num_epochs, self.steps)
        self.writer.add_scalar(
            'stats/mean_Q2', total_mean_q2/self.num_epochs, self.steps)
        self.writer.add_scalar(
            'stats/mean_entropy', total_mean_entropy/self.num_epochs,
            self.steps)

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        errors = torch.abs(curr_q1.detach() - target_q)
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        _, action_probs, log_action_probs, _ =\
            self.policy.sample(states)
        q1, q2 = self.critic(states)
        q = self.alpha * log_action_probs - torch.min(q1, q2)
        inside_term = torch.sum(action_probs * q, dim=1, keepdim=True)
        policy_loss = (inside_term * weights).mean()
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)
        return policy_loss, entropies

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def interval(self):
        if self.steps % self.eval_interval == 0:
            self.evaluate()
        if self.steps % self.memory_load_interval == 0:
            self.load_memory()
        if self.steps % self.model_save_interval == 0:
            self.save_weights()
            self.save_models()
        if self.steps % self.target_update_interval == 0:
            self.soft_update()

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward

        mean_return = np.mean(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('Learner \t '
              f'Num steps: {self.steps:<5} \t '
              f'reward: {mean_return:<5.1f}')

    def save_models(self):
        self.critic.save(
            os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))
        self.policy.save(
            os.path.join(self.model_dir, 'policy.pth'))