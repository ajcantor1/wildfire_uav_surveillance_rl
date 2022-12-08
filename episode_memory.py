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
    
    sample_episodes = random.sample(self.episodes, batch_size)

    min_step = min([len(episode) for episode in sample_episodes])

    return [deque(islice(episode, 0, min_step)) for episode in sample_episodes], min_step
    
  def __len__(self):
      return len(self.episodes)



