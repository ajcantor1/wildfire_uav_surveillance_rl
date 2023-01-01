import torch
import torch.nn as nn
from networks.basedqn import BaseDQN
from torch.distributions import Categorical
import torch.nn.functional as F

class PPONet(BaseDQN):

  def __init__(self, _device, _channels, _height, _width, _outputs):

    super().__init__(_device, _channels, _height, _width, _outputs)
    self.to(device=_device)
    self.fc1  = nn.Sequential(
      nn.Linear(5, 50),
      nn.ReLU(),
      nn.Linear(50, 50),
      nn.ReLU(),
      nn.Linear(50, 50),
      nn.ReLU(),
      nn.Linear(50, 50),
      nn.ReLU(),
      nn.Linear(50, 50),
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
      nn.Linear(150, 200),
      nn.ReLU(),
    )
    
    self.actor = nn.Linear(200, _outputs)

    self.critic = nn.Linear(256, 1)
    self._initialize_weights()


  def forward(self, belief_map, state_vector):

    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map),1)
    fc2_out = self.fc2(conv_out)
    
    fc3_out = self.fc3(torch.cat((fc1_out, fc2_out), dim=1))
    return self.actor(fc3_out), self.critic(fc3_out)

  def _initialize_weights(self):
    for module in self.modules():
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
        nn.init.constant_(module.bias, 0)



