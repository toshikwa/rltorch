import os
import time
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rltorch.q_function.conv import ConvQNetwork
from rltorch.env import make_pytorch_env
from rltorch.memory import PrioritizedMemory


def learner_process(env_id, log_dir, memory_queue, param_dict, seed):

    learner = Learner(env_id, log_dir, memory_queue, param_dict, seed)
    learner.run()


class Learner:

    def __init__(self, env_id, log_dir, memory_queue, param_dict, seed):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env = make_pytorch_env(env_id)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.net = ConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device)
        self.target_net = ConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device)

        self.batch_size = 64
        self.start_steps = 10000
        self.gamma = 0.99
        self.multi_step = 3
        self.gamma_n = self.gamma ** self.multi_step
        self.memory_size = 4e5
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.lr = 0.00025 / 4
        self.epochs = 32
        self.num_steps = 0

        self.target_net.load_state_dict(self.net.state_dict())
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)

        self.log_save_interval = 1
        self.memory_load_interval = 10
        self.model_save_interval = 10
        self.target_update_interval = 100
        self.eval_inderval = 100

        self.memory_queue = memory_queue
        self.memory = PrioritizedMemory(
            self.memory_size, self.env.observation_space.shape, (1,),
            self.device, self.gamma, multi_step=self.multi_step,
            is_image=True, alpha=self.per_alpha, beta=self.per_beta,
            beta_annealing=0.0)
        self.param_dict = param_dict
        self._save_params()

        self.log_dir = log_dir
        self.summary_path = os.path.join(log_dir, 'summary', 'leaner')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.writer = SummaryWriter(log_dir=self.summary_path)

    def run(self):
        while len(self.memory) <= self.start_steps:
            self._load_memory()

        self.time = time.time()
        while True:
            self.num_steps += 1
            self.learn()
            self.interval_process()

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)
        action_bar = np.zeros((self.env.action_space.n), np.int)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.action(state)
                next_state, reward, done, _ = self.env.step(action)
                action_bar[action] += 1
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'reward/test', mean_return, self.num_steps)
        print('********  '
              f'Num steps: {self.num_steps:<5} '
              f'reward: {mean_return:<5.1f}+/- {std_return:<5.1f}'
              ' ********')

    def action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_digits = self.net(state)
        action = action_digits.argmax().item()
        return action

    def interval_process(self):
        if self.num_steps % self.eval_inderval == 0:
            self.evaluate()

        if self.num_steps % self.memory_load_interval == 0:
            self._load_memory()

        if self.num_steps % self.model_save_interval == 0:
            self._save_params()

        if self.num_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def learn(self):
        total_loss = 0
        for epoch in range(self.epochs):
            batch, indices, weights = \
                self.memory.sample(self.batch_size)

            curr_q, target_q = self.calc_q(batch)
            loss = torch.mean((curr_q - target_q).pow(2) * weights)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            errors = torch.abs(curr_q.detach() - target_q).cpu().numpy()
            self.memory.update_priority(indices, errors)

            total_loss += loss.detach().item()

        if self.num_steps % self.log_save_interval == 0:
            self.writer.add_scalar(
                "loss/learner", total_loss / self.epochs,
                self.num_steps)
            print(
                f"Learer \t loss: {total_loss / self.epochs:< 8.3f} "
                f"memory: {len(self.memory):<5} \t")

    def calc_q(self, batch):
        state, action, reward, next_state, done = batch

        curr_q = self.net(state).gather(1, action.long()).view(-1, 1)

        with torch.no_grad():
            next_action = torch.argmax(
                self.net(next_state), 1).view(-1, 1)
            next_q = self.target_net(
                    next_state).gather(1, next_action.long()).view(-1, 1)

        target_q = reward + (1.0 - done) * self.gamma_n * next_q

        return curr_q, target_q

    def _save_params(self):
        self.param_dict['net'] = deepcopy(
            self.net).cpu().state_dict()
        self.param_dict['target_net'] = deepcopy(
            self.target_net).cpu().state_dict()

    def _load_memory(self):
        while not self.memory_queue.empty():
            batch = self.memory_queue.get()
            self.memory.load_memory(batch)

    def __del__(self):
        self.env.close()
        self.writer.close()
