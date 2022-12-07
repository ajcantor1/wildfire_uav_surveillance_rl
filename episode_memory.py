from collections import  deque
import random
from transition import Transition
from replay_memory import ReplayMemory

MAX_EPISODE_LENGTH = 200

class EpisodeMemory:

  def __init__(self, capicity=MAX_EPISODE_LENGTH):
    self._episode = ReplayMemory(capicity)

  def push(self, *args):
    self._episode.append(Transition(*args))

  @property
  def episode(self):
    return self._episode


  def __getitem__(self, index):
    return self._episode[int(index)]

  def __len__(self):
    return len(self._episode)

class EpisodeBuffer:
  def __init__(self, capicity=10000):
    self.capicity = capicity
    self.episodes = deque([],maxlen=self.capicity)

  def push(self, episode):
    self.episodes.push(episode)

  def sample(self, batch_size):
    
    sample_episodes = random.sample(self.episodes, batch_size)

    min_step = min([len(episode) for episode in sample_episodes])

    return [episode[:min_step] for episode in sample_episodes], min_step
    
  def __len__(self):
      return len(self.episodes)



