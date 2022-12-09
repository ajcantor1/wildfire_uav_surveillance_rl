import torch.nn as nn
import torch
from models.basedqn import BaseDQN
import math
import random

class DQN(BaseDQN):

  def __init__(self, _channels, _height, _width, _outputs):
    super().__init__(_channels, _height, _width, _outputs)

    self.fc1  = nn.Sequential(
      nn.Linear(5, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU()
    )

    self.conv = nn.Sequential(
      nn.Conv2d(2, 64, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2),
      nn.Conv2d(64, 64, kernel_size=3),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2)
    )
  
    conv_out_size = self._get_conv_out()

    self.fc2 = nn.Sequential(
      nn.Linear(conv_out_size, 500),
      nn.ReLU(),
      nn.Linear(500, 100),
      nn.ReLU(),
    )

    self.fc3 = nn.Sequential(
      nn.Linear(200, 200),
      nn.ReLU(),
      nn.Linear(200, 2),
    )


  def forward(self, belief_map, state_vector):
    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map),1)
    fc2_out = self.fc2(conv_out)
    fc3_out = self.fc3(torch.cat((fc1_out, fc2_out), dim=1))
    return fc3_out

  def select_action(self, belief_map, state_vector, steps):
    sample = random.random()
    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
      math.exp(-1. * steps / self.eps_decay)

    if sample > eps_threshold:
      with torch.no_grad():
        output = self(belief_map, state_vector).max(1)[1].view(1, 1)
        return output
    else:
      return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)