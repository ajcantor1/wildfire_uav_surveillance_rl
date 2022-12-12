from collections import deque
import random

class EpisodeBuffer:


  def __init__(self, capacity=50):
    self.capicity = capacity
    self.episodes = deque([],maxlen=self.capicity)

  def push(self, episode):
    self.episodes.append(episode)

  def sample(self, batch_size, episode_length):
    
    episode_batch = random.sample(self.episodes, batch_size)

    for i in range(batch_size):
        episode_start = random.randint(0, 60)
        episode_end  =  min(episode_start+episode_length, len(episode_batch[i]))
        episode_batch[i] = episode_batch[i][episode_start:episode_end]
    
    min_episode_length = min([len(episode) for episode in episode_batch])
    
    for i in range(batch_size):
      episode_batch[i] = episode_batch[i][0:min_episode_length]

    return episode_batch, min_episode_length

  def __len__(self):
      return len(self.episodes)