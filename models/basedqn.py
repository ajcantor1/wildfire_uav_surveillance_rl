import torch.nn as nn
import torch
import numpy as np

class BaseDQN(nn.Module):

  def _get_conv_out(self):
    o = self.conv(torch.zeros(1, (self.channels, self.height, self.width)))
    return int(np.prod(o.size()))

  @property
  def height(self):
    return self._height
  
  @property
  def width(self):
    return self._width

  @property
  def channels(self):
    return self._channels

  @property
  def _outputs(self):
    return self._outputs
    
  def __init__(self, _channels, _height, _width, _outputs):
    super().__init__()
    
    self._channels = _channels
    self._height = _height
    self._width = _width
    self._outputs = _outputs

  def forward(self, belief_map, state_vector):
    pass