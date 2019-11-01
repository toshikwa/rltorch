import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .multi_step import MultiStepMemory


class PrioritizedMemory(MultiStepMemory):

    def __init__(self, capacity, state_shape, action_shape, device,
                 gamma=0.99, multi_step=3, is_image=False, alpha=0.6,
                 beta=0.4, beta_annealing=0.001, epsilon=1e-4):
        super(PrioritizedMemory, self).__init__(
            capacity, state_shape, action_shape, device, gamma, multi_step,
            is_image)
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon
        self.priority = np.zeros(
            (self.capacity, 1), dtype=np.float32)

    def append(self, state, action, reward, next_state, done, error,
               episode_done=False):
        self.buff.append(state, action, reward)

        if len(self.buff) == self.multi_step:
            state, action, reward = self.buff.get(self.gamma)
            self.priority[self._p] = self.calc_priority(error)
            self._append(state, action, reward, next_state, done)

        if episode_done or done:
            self.buff.reset()

    def update_priority(self, indices, errors):
        self.priority[indices] = self.calc_priority(errors)

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        self.beta = min(1. - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(self.priority[:self._n], batch_size)
        indices = list(sampler)
        batch = self._sample(indices)

        p = self.priority[indices] / np.sum(self.priority[:self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)

        return batch, indices, weights
