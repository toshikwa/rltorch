import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .base import ApexAgent
from ...q_function import DiscreteConvQNetwork
from ...memory import PrioritizedMemory


class ApexLearner(ApexAgent):

    def __init__(self, env, log_dir, shared_memory, shared_weights,
                 batch_size=64, lr=0.00025/4, memory_size=4e5, gamma=0.99,
                 multi_step=3, alpha=0.4, num_epochs=3, start_steps=1000,
                 beta=0.6, beta_annealing=0.0, clip_grad=5.0, log_interval=10,
                 memory_load_interval=5, model_save_interval=5,
                 target_update_interval=100, eval_interval=1000, cuda=True,
                 seed=0):
        self.env = env
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.shared_memory = shared_memory
        self.shared_weights = shared_weights

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.net = DiscreteConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device)
        self.target_net = DiscreteConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.save_weights()

        self.memory = PrioritizedMemory(
            memory_size, self.env.observation_space.shape, (1,),
            self.device, gamma, multi_step, is_image=True,
            alpha=alpha, beta=beta, beta_annealing=beta_annealing)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary', 'leaner')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.steps = 0
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.num_epochs = num_epochs
        self.clip_grad = clip_grad
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
        total_loss = 0.
        total_mean_grads = 0.
        total_mean_q = 0.

        for epoch in range(self.num_epochs):
            batch, indices, weights = \
                self.memory.sample(self.batch_size)

            curr_q = self.calc_current_q(*batch)
            target_q = self.calc_target_q(*batch)
            loss = torch.mean((curr_q - target_q).pow(2) * weights)

            total_mean_grads += self.update_params(
                self.optim, self.net, loss, self.clip_grad)

            errors = torch.abs(curr_q.detach() - target_q).cpu().numpy()
            self.memory.update_priority(indices, errors)

            total_loss += loss.detach().item()
            total_mean_q += curr_q.detach().mean().item()

        if self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                "loss/learner", total_loss / self.num_epochs,
                self.steps)
            self.writer.add_scalar(
                "stats/mean_grads", total_mean_grads / self.num_epochs,
                self.steps)
            self.writer.add_scalar(
                "stats/mean_Q", total_mean_q / self.num_epochs,
                self.steps)
            print(
                f"Learer \t loss: {total_loss / self.num_epochs:< 8.3f} "
                f"memory: {len(self.memory):<5} \t")

    def interval(self):
        if self.steps % self.eval_interval == 0:
            self.evaluate()
        if self.steps % self.memory_load_interval == 0:
            self.load_memory()
        if self.steps % self.model_save_interval == 0:
            self.save_weights()
            self.save_models()
        if self.steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)
        action_bar = np.zeros((self.env.action_space.n), np.int)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                action_bar[action] += 1
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('********  '
              f'Num steps: {self.steps:<5} '
              f'reward: {mean_return:<5.1f}+/- {std_return:<5.1f}'
              ' ********')

    def save_models(self):
        self.net.save(
            os.path.join(self.model_dir, 'net.pth'))
        self.target_net.save(
            os.path.join(self.model_dir, 'target_net.pth'))
