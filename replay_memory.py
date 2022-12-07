from collections import  deque
import random
from transition import Transition

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