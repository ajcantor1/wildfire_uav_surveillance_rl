import os

class BasePolicy(object):

  def __init__(self, _device):
    self._device = _device

  @property
  def device(self):
    return self._device

  @device.setter
  def device(self, _device):
    self._device = _device

    