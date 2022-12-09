from collections import deque
import random
import numpy as np
import torch

MAX_EPISODE_LENGTH = 200

class EpisodeMemory():
  """Episode memory for recurrent agent"""

  def __init__(self, random_update=False, max_epi_num=100, max_epi_len=200, lookup_step=None):
    self.random_update = random_update # if False, sequential update
    self.max_epi_num = max_epi_num
    self.max_epi_len = max_epi_len
    self.lookup_step = lookup_step


    self.memory = deque(maxlen=self.max_epi_num)

  def put(self, episode):
    self.memory.append(episode)

  def sample(self, batch_size):
    sampled_buffer = []

    ##################### RANDOM UPDATE ############################
    if self.random_update: # Random upodate
      sampled_episodes = random.sample(self.memory, batch_size)
      

      min_step = self.max_epi_len

      for episode in sampled_episodes:
        min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

      for episode in sampled_episodes:
        if min_step > self.lookup_step: # sample buffer with lookup_step size
          idx = np.random.randint(0, len(episode)-self.lookup_step+1)
          sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
          sampled_buffer.append(sample)
        else:
          idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
          sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
          sampled_buffer.append(sample)

    ##################### SEQUENTIAL UPDATE ############################           
    else: # Sequential update
      idx = np.random.randint(0, len(self.memory))
      sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

    return torch.cat(sampled_buffer), len(sampled_buffer[0]['observation']) # buffers, sequence_length

  def __len__(self):
      return len(self.memory)


class EpisodeBuffer:


  def __init__(self):
    self.observation = []
    self.state = []
    self.action = []
    self.next_observation = []
    self.next_state = []
    self.reward = []


  def put(self, *transition):
    self.observation.append(transition[0])
    self.state.append(transition[1])
    self.action.append(transition[2])
    self.next_observation.append(transition[3])
    self.next_state.append(transition[4])
    self.reward.append(transition[5])
        
  def sample(self, random_update=False, lookup_step=None, idx=None):

    return dict(
      observation=torch.cat(self.observation),
      state=torch.cat(self.state),
      action=torch.cat(self.action),
      next_observation=torch.cat(self.next_observation),
      next_state=torch.cat(self.next_state),
      reward=torch.cat(self.reward)
    )

  def __len__(self):
      return len(self.observation)