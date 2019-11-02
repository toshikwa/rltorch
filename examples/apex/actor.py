import os
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter

from rltorch.q_function.conv import ConvQNetwork
from rltorch.env import make_pytorch_env
from rltorch.memory import SharedPrioritizedMemory


def actor_process(env_id, log_dir, memory_queue, param_dict, actor_id,
                  num_actors, seed):

    actor = Actor(env_id, log_dir, memory_queue, param_dict, actor_id,
                  num_actors, seed)
    actor.run()


class Actor:
    space_size = 55

    def __init__(self, env_id, log_dir, memory_queue, param_dict, actor_id,
                 num_actors, seed):
        self.env = make_pytorch_env(env_id)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.net = ConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device).eval()
        self.target_net = ConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device).eval()
        self.target_net.load_state_dict(self.net.state_dict())

        self.actor_id = actor_id
        self.gamma = 0.99
        self.multi_step = 3
        self.gamma_n = self.gamma ** self.multi_step
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.memory_size = 1e4

        self.epsilon = 0.4 ** (1 + actor_id * 7 / (num_actors-1))\
            if num_actors > 1 else 0.4
        self.num_episodes = 0
        self.num_steps = 0
        self.log_save_interval = 20
        self.memory_save_interval = 10
        self.model_load_interval = 5
        self.summary_save_interval = 1

        self.log_dir = log_dir
        self.summary_path = os.path.join(
            log_dir, 'summary', f'actor-{self.actor_id}')

        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.writer = SummaryWriter(log_dir=self.summary_path)

        self.memory = SharedPrioritizedMemory(
            memory_queue, 10000, self.env.observation_space.shape, (1,),
            self.gamma, self.multi_step, is_image=True,
            alpha=self.per_alpha, beta=self.per_beta,
            beta_annealing=0.0)
        self.param_dict = param_dict
        self.num_actions = self.env.action_space.n
        self.max_episode_steps = self.env.spec.tags.get(
            'wrapper_config.TimeLimit.max_episode_steps')

    def run(self):
        self.time = time.time()
        while True:
            self.num_episodes += 1
            self.act_episode()
            self.interval_process()

    def act_episode(self):
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode_steps += 1
            episode_reward += reward
            self.num_steps += 1

            # ignore the "done" if it comes from hitting the time horizon
            masked_done = False if episode_steps >= self.max_episode_steps\
                else done

            curr_q, target_q = self.calc_q(
                state, action, reward, next_state, masked_done)
            error = torch.abs(curr_q - target_q).item()
            self.memory.append(
                np.array(state, np.float32), action, reward,
                np.array(next_state, np.float32), masked_done, error,
                episode_done=done)

            state = next_state

        if self.num_episodes % self.log_save_interval == 0:
            self.writer.add_scalar(
                'reward/train', episode_reward, self.num_steps)

        print(' '*self.space_size,
              f'Actor {self.actor_id:<2} \t'
              f'Episode: {self.num_episodes:<4} \t'
              f'episode steps: {episode_steps:<4} \t'
              f'reward: {episode_reward:<5.1f}')

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.action(state)

    def action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_digits = self.net(state)
        action = action_digits.argmax().item()
        return action

    def interval_process(self):
        if self.num_episodes % self.model_load_interval == 0:
            self._load_params()
        if self.num_episodes % self.memory_save_interval == 0:
            self.memory.save()

    def _load_params(self):
        self.net.load_state_dict(self.param_dict['net'])
        self.target_net.load_state_dict(self.param_dict['target_net'])

    def calc_q(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor([action]).unsqueeze(-1).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(-1).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            curr_q = self.net(state).gather(1, action.long()).view(-1, 1)

        with torch.no_grad():
            next_action = torch.argmax(self.net(next_state), 1).view(-1, 1)
            next_q = self.target_net(
                next_state).gather(1, next_action.long()).view(-1, 1)

        target_q = reward + (1.0 - done) * self.gamma_n * next_q

        return curr_q, target_q

    def __del__(self):
        self.writer.close()
        self.env.close()
