import torch.nn as nn
import torch
from models.basedqn import BaseDQN

class DRQN(BaseDQN):

  def __init__(self, _channels, _height, _width, _outputs, _hidden_space = 200):
    super().__init__(_channels, _height, _width, _outputs)

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


  def forward(self, belief_map, state_vector, hidden_state, cell_state):
    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map), 1)
    fc2_out = self.fc2(conv_out)
    ltsm_out, (new_hidden_state, new_cell_state) = self.ltsm(torch.cat((fc1_out, fc2_out), dim=1), (hidden_state, cell_state))
    fc3_out = self.fc3(ltsm_out)
    return fc3_out, new_hidden_state, new_cell_state


  def init_hidden_state(self, training=None, batch_size=1):

    if training is True:
        return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
    else:
        return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])