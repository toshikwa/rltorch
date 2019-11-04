import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .base import SacDiscreteAgent
from rltorch.memory import Memory, MultiStepMemory, PrioritizedMemory
from rltorch.policy import ConvCategoricalPolicy
from rltorch.q_function import TwinedDiscreteConvQNetwork
from rltorch.agent import to_batch, hard_update


class SacDiscreteActor(SacDiscreteAgent):
    space_size = 55

    def __init__(self, env, log_dir, shared_memory, shared_weights,
                 actor_id, num_actors=1, memory_size=1e4, gamma=0.99,
                 multi_step=3, per=True, alpha=0.6, beta=0.4,
                 beta_annealing=0.001, start_steps=10000, log_interval=10,
                 memory_save_interval=5, model_load_interval=5, cuda=True,
                 seed=0):

        self.actor_id = actor_id
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
            self.env.action_space.n).to(self.device).eval()
        self.critic = TwinedDiscreteConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device).eval()
        self.critic_target = TwinedDiscreteConvQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n).to(self.device).eval()
        hard_update(self.critic_target, self.critic)

        if per:
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                (1,), self.device, gamma, multi_step,
                is_image=True, alpha=alpha, beta=beta,
                beta_annealing=beta_annealing)
        elif multi_step == 1:
            self.memory = Memory(
                memory_size, self.env.observation_space.shape,
                (1,), self.device, is_image=True)
        else:
            self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                (1,), self.device, gamma, multi_step,
                is_image=True)

        self.log_dir = log_dir
        self.summary_dir = os.path.join(
            log_dir, 'summary', f'actor-{self.actor_id}')
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = SummaryWriter(log_dir=self.summary_dir)

        self.episodes = 0
        self.steps = 0
        self.per = per
        self.multi_step = multi_step
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.log_interval = log_interval
        self.memory_save_interval = memory_save_interval
        self.model_load_interval = model_load_interval

        load = False
        while load is False:
            load = self.load_weights()

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

            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            clipped_reward = max(min(reward, 1.0), -1.0)

            if self.per:
                batch = to_batch(
                    state, action, clipped_reward,
                    next_state, masked_done, self.device)
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()

                self.memory.append(
                    state, action, clipped_reward, next_state,
                    masked_done, error, episode_done=done)
            elif self.multi_step == 1:
                self.memory.append(
                    state, action, clipped_reward, next_state, masked_done)
            else:
                self.memory.append(
                    state, action, clipped_reward, next_state, masked_done,
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
