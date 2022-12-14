from collections import deque
import random

class EpisodeBuffer:


  def __init__(self, capacity=200):
    self.capicity = capacity
    self.episodes = deque([],maxlen=self.capicity)

  def push(self, episode):
    self.episodes.append(episode)

  def sample(self, episode_length):
    
    episode_batch = self.episodes[random.randint(0, len(self.episodes)-1)]

    episode_start = random.randint(0, 50)
    episode_end  =  min(episode_start+episode_length, len(episode_batch))
    episode_batch = episode_batch[episode_start:episode_end]
    
    return episode_batch, episode_end - episode_start

  def __len__(self):
    return len(self.episodes)