import torch.nn as nn
import torch
import numpy as np
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 200000

class BaseDQN(nn.Module):

  def _get_conv_out(self):
    o = self.conv(torch.zeros(1, self.channels, self.height, self.width))
    return int(np.prod(o.size()))

  @property
  def height(self):
    return self._height

  @height.setter
  def height(self, _height):
    self._height = _height
  
  @property
  def width(self):
    return self._width

  @width.setter
  def width(self, _width):
    self._width = _width

  @property
  def channels(self):
    return self._channels

  @channels.setter
  def channels(self, _channels):
    self._channels = _channels

  @property
  def outputs(self):
    return self._outputs

  @outputs.setter
  def outputs(self, _outputs):
    self._outputs = _outputs

  @property
  def device(self):
    return self._device

  @device.setter
  def device(self, _device):
    self._device = _device

  @property
  def eps_start(self):
    return self._eps_start

  @eps_start.setter
  def device(self, _eps_start):
    self._eps_start = _eps_start

  @property
  def eps_end(self):
    return self._eps_end

  @eps_end.setter
  def device(self, _eps_end):
    self._eps_end = _eps_end

  @property
  def eps_decay(self):
    return self._eps_decay

  @eps_decay.setter
  def device(self, _eps_decay):
    self._eps_decay = _eps_decay
    
  def __init__(self, _device, _channels, _height, _width, _outputs, _eps_start=EPS_START, _eps_end=EPS_END, _eps_decay=EPS_DECAY):
    super().__init__()
    
    self._channels = _channels
    self._height = _height
    self._width = _width
    self._outputs = _outputs
    self._device = _device

    self._eps_start = _eps_start
    self._eps_end = _eps_end
    self._eps_decay = _eps_decay

  def forward(self, belief_map, state_vector):
    pass

  def select_action(self, belief_map, state_vector, steps):
    pass