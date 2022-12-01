from abc import ABCMeta, abstractmethod
import numpy as np
class AbstractFireEnv(metaclass = ABCMeta):

  def __init__(self, _height, _width):
    self._height = _height
    self._width  = _width
    self._time_steps = 0
    self._observation = None

  @property
  def height(self):
    return self._height
  
  @property
  def width(self):
    return self._width

  @property 
  def time_steps(self):
    return self._time_steps

  @time_steps.setter
  def time_steps(self, _time_steps):
    self._time_steps = _time_steps

  @property
  def observation(self):
    return self._observation

  @observation.setter
  def observation(self, _observation):
    self._observation = _observation

  def step(self):
    self._time_steps += 1
    self.observation = self.next_observation()
    return self.observation

  def plot_heat_map(self, fig, ax):
    ax.axis(xmin=0, xmax=self._width)
    ax.axis(ymin=0, ymax=self._height)  
    heat_map_plot = ax.imshow(self.observation, cmap='hot')
    return heat_map_plot

  def reset(self):
    self._time_steps = 0
    self.observation = self.reset_observation()
    seed = self.observation.copy()
    #for _ in range(30):
    #  self.step()
    return seed

  def fire_in_range(self,margin=2):
    burnX, burnY = np.where(self.observation==1)
    return min(burnX)>=margin and min(burnY)>=margin and max(burnX)<=99-margin and max(burnY)<=99-margin

  @abstractmethod
  def next_observation(self):
    pass

  @abstractmethod
  def reset_observation(self):
    pass