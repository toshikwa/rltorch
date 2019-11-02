import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .base import ApexAgent
from ...env.atari_wrappers import make_pytorch_env
from ...memory import PrioritizedMemory
from ...q_function import ConvQNetwork


class ApexActor(ApexAgent):
    space_size = 55

    def __init__(self, env_id, log_dir, shared_memory, shared_weights,
                 actor_id, num_actors, memory_size=1e4, gamma=0.99,
                 multi_step=3, alpha=0.4, beta=0.6, beta_annealing=0.0,
                 log_interval=10, memory_save_interval=5,
                 model_load_interval=5, cuda=True, seed=0):

        self.actor_id = actor_id
        self.env = make_pytorch_env(env_id)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.shared_memory = shared_memory
        self.shared_weights = shared_weights

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.net = ConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device).eval()
        self.target_net = ConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device).eval()
        self.target_net.load_state_dict(self.net.state_dict())

        self.memory = PrioritizedMemory(
            memory_size, self.env.observation_space.shape, (1,),
            self.device, gamma, multi_step, is_image=True,
            alpha=alpha, beta=beta, beta_annealing=beta_annealing)

        self.log_dir = log_dir
        self.summary_path = os.path.join(
            log_dir, 'summary', f'actor-{self.actor_id}')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.writer = SummaryWriter(log_dir=self.summary_path)

        if num_actors > 1:
            self.epsilon = 0.4 ** (1 + actor_id * 7 / (num_actors-1))
        else:
            self.epsilon = 0.4

        self.episodes = 0
        self.steps = 0
        self.gamma_n = gamma ** multi_step
        self.log_interval = log_interval
        self.memory_save_interval = memory_save_interval
        self.model_load_interval = model_load_interval

    def run(self):
        while True:
            self.episodes += 1
            self.act_episode()
            self.interval()

    def act_episode(self):
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore 'done' when hitting the time horizon
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            batch = self.to_batch(
                state, action, reward, next_state, masked_done)
            with torch.no_grad():
                curr_q = self.calc_current_q(*batch)
            target_q = self.calc_target_q(*batch)
            error = torch.abs(curr_q - target_q).item()

            self.memory.append(
                state, action, reward, next_state, masked_done, error,
                episode_done=done)

            state = next_state

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', episode_reward, self.steps)

        print(' '*self.space_size,
              f'Actor {self.actor_id:<2} \t'
              f'episode: {self.episodes:<4} \t'
              f'episode steps: {episode_steps:<4} \t'
              f'reward: {episode_reward:<5.1f}')

    def interval(self):
        if self.episodes % self.model_load_interval == 0:
            self.load_weights()
        if self.episodes % self.memory_save_interval == 0:
            self.save_memory()
