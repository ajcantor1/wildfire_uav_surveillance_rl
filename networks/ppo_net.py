import torch
import torch.nn as nn
from networks.basedqn import BaseDQN
from torch.distributions import Categorical
import torch.nn.functional as F
class PPONet(BaseDQN):

  def __init__(self, _device, _channels, _height, _width, _outputs):

    super().__init__(_device, _channels, _height, _width, _outputs)

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

    

    self.actor = nn.Linear(200, _outputs)

    self.critc = nn.Linear(200, 1)

  def forward(self, belief_map, state_vector):

    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map),1)
    fc2_out = self.fc2(conv_out)
    flattened_concatenated = torch.cat((fc1_out, fc2_out), dim=1)
    return F.softmax(self.actor(flattened_concatenated), dim=1), self.critic(flattened_concatenated)


