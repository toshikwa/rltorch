from .base import Memory
from .multi_step import MultiStepMemory
from .prioritized import PrioritizedMemory


class SharedMemory:

    def __init__(self, queue, capacity, state_shape, action_shape,
                 is_image=False):
        self.memory = Memory(
            capacity, state_shape, action_shape, None, is_image)
        self.queue = queue

    def append(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def save(self):
        if len(self.memory) > 0:
            self.queue.put(self.memory.get())
            self.memory.reset()


class SharedMultiStepMemory(SharedMemory):

    def __init__(self, queue, capacity, state_shape, action_shape,
                 gamma=0.99, multi_step=3, is_image=False):
        self.memory = MultiStepMemory(
            capacity, state_shape, action_shape, gamma, multi_step, is_image)
        self.queue = queue

    def append(self, state, action, reward, next_state, done,
               episode_done=False):
        self.memory.append(
            state, action, reward, next_state, done, episode_done)


class SharedPrioritizedMemory(SharedMemory):

    def __init__(self, queue, capacity, state_shape, action_shape,
                 gamma=0.99, multi_step=3, is_image=False, alpha=0.6,
                 beta=0.4, beta_annealing=0.001, epsilon=1e-4):
        self.memory = PrioritizedMemory(
            capacity, state_shape, action_shape, None,
            gamma, multi_step, is_image, alpha,
            beta, beta_annealing, epsilon)
        self.queue = queue

    def append(self, state, action, reward, next_state, done, error,
               episode_done=False):
        self.memory.append(
            state, action, reward, next_state, done, error, episode_done)
