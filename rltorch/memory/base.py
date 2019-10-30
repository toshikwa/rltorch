import numpy as np
import torch


class Memory:

    def __init__(self, capacity, state_shape, action_shape, device,
                 is_image=False):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.is_image = is_image
        self.state_type = np.uint8 if self.is_image else np.float32

        self._n = 0
        self._p = 0

        self.states = np.zeros(
            (self.capacity, *state_shape), dtype=self.state_type)
        self.actions = np.zeros(
            (self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros(
            (self.capacity, *state_shape), dtype=self.state_type)
        self.dones = np.zeros(
            (self.capacity, 1), dtype=np.float32)

    def append(self, state, action, reward, next_state, done):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        if self.is_image:
            state = (state*255).astype(np.uint8)
            next_state = (next_state*255).astype(np.uint8)

        self.states[self._p] = state
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.next_states[self._p] = next_state
        self.dones[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        if self.is_image:
            states = self.states[indices].astype(np.float32) / 255.
            next_states = self.next_states[indices].astype(np.float32) / 255.
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n
