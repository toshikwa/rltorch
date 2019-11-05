import os
import numpy as np
import torch
import torch.optim as optim
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from .base import SacAgent
from rltorch.memory import MultiStepMemory, PrioritizedMemory
from rltorch.policy import LinearGaussianPolicy
from rltorch.q_function import TwinnedContinuousLinearQNetwork
from rltorch.agent import soft_update, hard_update, update_params


class SacLearner(SacAgent):

    def __init__(self, env, log_dir, shared_memory, shared_weights,
                 batch_size=256, lr=0.0003, hidden_units=[256, 256],
                 memory_size=1e6, gamma=0.99, tau=0.005, entropy_tuning=True,
                 ent_coef=0.2, multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.001, grad_clip=None, update_per_steps=1,
                 start_steps=10000, log_interval=10, memory_load_interval=5,
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

        self.policy = LinearGaussianPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic = TwinnedContinuousLinearQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic_target = TwinnedContinuousLinearQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device).eval()

        hard_update(self.critic_target, self.critic)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = torch.tensor(ent_coef).to(self.device)

        self.save_weights()
        if per:
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary', 'leaner')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.steps = 0
        self.epochs = 0
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.update_per_steps = update_per_steps
        self.log_interval = log_interval
        self.memory_load_interval = memory_load_interval
        self.model_save_interval = model_save_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

    def run(self):
        while len(self.memory) <= self.start_steps:
            self.load_memory()

        while True:
            self.epochs += 1
            for _ in range(self.update_per_steps):
                self.steps += 1
                self.learn()
            self.interval()

    def learn(self):
        if self.per:
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            self.update_params(
                self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        if self.per:
            self.memory.update_priority(indices, errors.cpu().numpy())

        if self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(), self.steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(), self.steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(), self.steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(), self.steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(), self.steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(), self.steps)

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
        sampled_action, entropy, _ = self.policy.sample(states)
        q1, q2 = self.critic(states, sampled_action)
        q = torch.min(q1, q2)
        policy_loss = torch.mean((- q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def interval(self):
        if self.epochs % self.eval_interval == 0:
            self.evaluate()
        if self.epochs % self.memory_load_interval == 0:
            self.load_memory()
        if self.epochs % self.model_save_interval == 0:
            self.save_weights()
            self.save_models()
        if self.epochs % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

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
        print('Learner '
              f'Num steps: {self.steps:<5} \t '
              f'reward: {mean_return:<5.1f}')

    def save_models(self):
        self.critic.save(
            os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))
        self.policy.save(
            os.path.join(self.model_dir, 'policy.pth'))
