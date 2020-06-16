import numpy as np
from random import uniform


class PriorityExperienceBuffer:

    def __init__(self, capacity):
        self.current_idx = 0
        self.current_len = 0
        self._capacity = capacity
        self._tree = np.zeros((capacity * 2) - 1)
        self._data = np.zeros(capacity, dtype=object)

    def add(self, prob, data):
        idx = self.current_idx + self._capacity - 1

        self._data[self.current_idx] = data
        self.update_tree(idx, prob)

        self.current_idx += 1
        if self.current_len < self._capacity:
            self.current_len += 1

        if self.current_idx >= self._capacity:
            self.current_idx = 0

    def update_tree(self, idx, prob):
        change = prob - self._tree[idx]
        self._tree[idx] = prob
        self._propagate_change(idx, change)

    def _propagate_change(self, idx, change):
        parent_idx = (idx - 1) // 2
        self._tree[parent_idx] += change

        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def total(self):
        return self._tree[0]

    def __len__(self):
        return self.current_len

    def _retrieve_sample(self, node_id, rand_state):
        left_node_id = 2 * node_id + 1
        right_node_id = left_node_id + 1
        if left_node_id >= len(self._tree):
            return node_id

        if rand_state <= self._tree[left_node_id]:
            return self._retrieve_sample(left_node_id, rand_state)
        else:
            return self._retrieve_sample(right_node_id, rand_state - self._tree[left_node_id])

    def _get(self, rand_state):
        tree_idx = self._retrieve_sample(0, rand_state)

        data_idx = tree_idx - self._capacity - 1
        if self._data[data_idx] == 0:
            return self._get(uniform(0.1, self.total()))

        return tree_idx, self._data[data_idx]

    def sample(self, batch_size):
        samples = []
        idxs = []
        for i in range(batch_size):
            rand_state = uniform(0.00000001, self.total())
            tr_idx, sample = self._get(rand_state)
            samples.append(sample)
            idxs.append(tr_idx)

        return idxs, samples
