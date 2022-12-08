from collections import deque
import random
import numpy as np

MAX_EPISODE_LENGTH = 200

class EpisodeMemory():
  """Episode memory for recurrent agent"""

  def __init__(self, random_update=False, max_epi_num=100, max_epi_len=200, batch_size=1, lookup_step=None):
    self.random_update = random_update # if False, sequential update
    self.max_epi_num = max_epi_num
    self.max_epi_len = max_epi_len
    self.batch_size = batch_size
    self.lookup_step = lookup_step


    self.memory = deque(maxlen=self.max_epi_num)

  def put(self, episode):
    self.memory.append(episode)

  def sample(self):
    sampled_buffer = []

    ##################### RANDOM UPDATE ############################
    if self.random_update: # Random upodate
      sampled_episodes = random.sample(self.memory, self.batch_size)
      

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

    return sampled_buffer, len(sampled_buffer[0]['observation']) # buffers, sequence_length

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
    observation = np.array(self.observation)
    state = np.array(self.state)
    action = np.array(self.action)
    next_observation = np.array(self.next_observation)
    next_state = np.array(self.next_state)
    reward = np.array(self.reward)

    if random_update is True:
      observation  = observation[idx:idx+lookup_step]
      state  = state[idx:idx+lookup_step]
      action = action[idx:idx+lookup_step]
      next_observation = next_observation[idx:idx+lookup_step]
      next_state = next_state[idx:idx+lookup_step]
      reward = reward[idx:idx+lookup_step]

    return dict(
      observation=observation,
      state=state,
      action=action,
      next_observation=next_observation,
      next_state=next_state,
      reward=reward
    )

  def __len__(self):
      return len(self.observation)