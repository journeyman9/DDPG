import numpy as np
import collections
import random

class ReplayBuffer:
    def __init__(self, MAX_BUFFER, BATCH_SIZE):
        ''' right side of deque contains the most recent experiences '''
        self.max_buffer = MAX_BUFFER
        self.batch_size = BATCH_SIZE
        self.count = 0
        self.buffer = collections.deque()

    def add_sample(self, sample):
        if self.count < self.max_buffer:
            self.buffer.append(sample)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(sample)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        return batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
