from collections import deque
import random

class EpisodeBuffer:


  def __init__(self, capacity=20):
    self.capicity = capacity
    self.episodes = deque([],maxlen=self.capicity)

  def push(self, episode):
    self.episodes.append(episode)

  def sample(self, batch_size, episode_start, episode_end):
    
    episode_batch = random.sample(self.episodes, batch_size)
    
    min_episode_length = min([len(episode) for episode in episode_batch] + [episode_end-episode_start])
    
    for i in range(batch_size):
      episode_batch[i] = episode_batch[i][episode_start:episode_start+min_episode_length]

    return episode_batch, min_episode_length

  def __len__(self):
      return len(self.episodes)