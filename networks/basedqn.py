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

    
  def __init__(self, _device, _channels, _height, _width, _outputs):
    super().__init__()
    
    self._channels = _channels
    self._height = _height
    self._width = _width
    self._outputs = _outputs
    self._device = _device

    self.to(device=_device)
  



  def forward(self, belief_map, state_vector):
    pass

