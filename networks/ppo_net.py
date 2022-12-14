import torch
import torch.nn as nn
from networks.basedqn import BaseDQN
class IndependentPPOActor(BaseDQN):


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

    self.rnn_hidden_dim = _hidden_dim
    self.action_dim = _outputs
    self.fc3 = nn.Linear(200, self.rnn_hidden_dim)
    self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
    self.fc4 = nn.Linear(self.rnn_hidden_dim, self.action_dim)

  def forward(self, belief_map, state_vector, hidden):

    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map), 1)
    fc2_out = self.fc2(conv_out)
  
    fc3_out = torch.relu(self.fc3(torch.cat((fc1_out, fc2_out), dim=1)))
    rnn_out = self.rnn(fc3_out, hidden)

    fc4_out = torch.sigmoid(self.fc4(rnn_out))
    return fc4_out, rnn_out


class IndependentPPOCritic(BaseDQN):

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

    self.fc3 = nn.Sequential(
      nn.Linear(200, 200),
      nn.ReLU(),
      nn.Linear(200, 1),
      nn.ReLU()
    )

  def forward(self, belief_map, state_vector):
    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map),1)
    fc2_out = self.fc2(conv_out)
    fc3_out = self.fc3(torch.cat((fc1_out, fc2_out), dim=1))
    return fc3_out