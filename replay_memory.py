from collections import namedtuple, deque
import random
Transition = namedtuple('Transition',('belief_map', 'state_vector', 'action', 'next_belief_map', 'next_state_vector', 'reward'))
class ReplayMemory:

  def __init__(self, capacity):
    self.capicity = capacity
    self.memory = deque([],maxlen=self.capicity)

  def push(self, *args):
    """Save a transition"""
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)