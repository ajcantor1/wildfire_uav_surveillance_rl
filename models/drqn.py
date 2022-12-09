import torch.nn as nn
import torch
from models.basedqn import BaseDQN
import random
import math

class DRQN(BaseDQN):

  def __init__(self, _device, _channels, _height, _width, _outputs, _hidden_space = 200):
    super().__init__(_device, _channels, _height, _width, _outputs)

    self.hidden_space = _hidden_space

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

    self.ltsm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)

    self.fc3 = nn.Sequential(
      nn.Linear(self.hidden_space, self.hidden_space),
      nn.ReLU(),
      nn.Linear(self.hidden_space, 2),
    )


  def forward(self, belief_map, state_vector, hidden=None):

    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map), 1)
    fc2_out = self.fc2(conv_out)
    new_hidden = None
    ltsm_out = None
    if hidden is None:
      ltsm_out, new_hidden = self.ltsm(torch.cat((fc1_out, fc2_out), dim=1))
    else:
      ltsm_out, new_hidden = self.ltsm(torch.cat((fc1_out, fc2_out), dim=1), hidden)
    fc3_out = self.fc3(ltsm_out)
    return fc3_out, new_hidden

  def select_action(self, belief_map, state_vector, steps, hidden=None):
    sample = random.random()
    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
      math.exp(-1. * steps / self.eps_decay)
    
    with torch.no_grad():
      output = None
      new_hidden = None 
      if hidden is None:
        output, new_hidden = self(belief_map, state_vector)
      else:
        output, new_hidden = self(belief_map, state_vector, hidden)
      if sample > eps_threshold:
        return output.max(1)[1].view(1, 1), new_hidden
      else:
        return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long), new_hidden