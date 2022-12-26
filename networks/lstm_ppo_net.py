import torch
import torch.nn as nn
from networks.ppo_net import PPONet
from torch.distributions import Categorical
import torch.nn.functional as F

class LSTMPPONet(PPONet):

  def __init__(self, _device, _channels, _height, _width, _outputs):

    super().__init__(_device, _channels, _height, _width, _outputs)
    
    self.lstm = nn.LSTM(200, 200, batch_first=True)


  def forward(self, belief_map, state_vector, hidden = None, sequence_length=1):

    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map),1)
    fc2_out = self.fc2(conv_out)

    concatenated = torch.cat((fc1_out, fc2_out), dim=1)
    
    lstm_out = None
    hidden_out = None
    if hidden is None:
      hidden = self._init_recurrent_cell_states(sequence_length)
      lstm_out, hidden_out = self.lstm(concatenated, hidden)
    else:
      lstm_out, hidden_out = self.lstm(concatenated, hidden)

    fc3_out = self.fc3(lstm_out)

    return self.actor(fc3_out), self.critic(fc3_out), hidden_out

  def _init_recurrent_cell_states(self, num_sequences):

    num_sequences = 1 if num_sequences is None else num_sequences

    hidden = torch.zeros(1, 200, dtype=torch.float32, device=self.device).unsqueeze(0)
    cell = torch.zeros(1, 200, dtype=torch.float32, device=self.device).unsqueeze(0)
    return hidden, cell