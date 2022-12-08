from collections import deque
import random
from transition import Transition
from itertools import islice

MAX_EPISODE_LENGTH = 200

class EpisodeMemory:

  def __init__(self, capicity=MAX_EPISODE_LENGTH):
    self._episode = deque([],maxlen=capicity)

  def push(self, *args):
    self._episode.append(Transition(*args))

  def __getitem__(self, index):
    return self._episode[index]

  def __len__(self):
    return len(self._episode)

class EpisodeBuffer:
  def __init__(self, capicity=10000):
    self.capicity = capicity
    self.episodes = deque([],maxlen=self.capicity)

  def push(self, episode):
    self.episodes.append(episode)

  def sample(self, batch_size):
    
    sample_buffer = []

    sample_episodes = random.sample(self.episodes, batch_size)

    min_step = min([len(episode) for episode in sample_episodes])

    for episode in sample_episodes:
      for i in range(min_step):
        sample_buffer.append(episode[i])
    
    return sample_buffer
    
  def __len__(self):
      return len(self.episodes)



