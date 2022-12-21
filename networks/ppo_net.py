import torch
import torch.nn as nn
from networks.basedqn import BaseDQN

class PPONet(BaseDQN):

  def __init__(self, _device, _channels, _height, _width, _outputs, _hidden_dim = 200):

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

    self.ltsm = nn.LSTM(_hidden_dim, _hidden_dim, batch_first=True)

    self.fc3 = nn.Sequential(
      nn.Linear(_hidden_dim, _outputs),
      nn.ReLU()
    )


  def forward(self, belief_map, state_vector, hidden = None):

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

