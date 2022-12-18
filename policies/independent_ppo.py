from policies.base_policy import BasePolicy
from networks.ppo_net import IndependentPPOActor, IndependentPPOCritic
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class IndependentPPO(BasePolicy):

  def __init__(self, _device, _channels, _height, _width, _outputs):
    super().__init__(_device)

    self.ppo_actor = IndependentPPOActor(_device, _channels, _height, _width, _outputs).to(_device)
    self.ppo_critic = IndependentPPOCritic(_device, _channels, _height, _width, 1).to(_device)

    self.optimizer_actor = torch.optim.Adam(params=self.ppo_actor.parameters(), lr=0.001)
    self.optimizer_critic = torch.optim.Adam(params=self.ppo_critic.parameters(), lr=0.001)

	  self.cov_var = torch.full(size=(_outputs,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

  def init_hidden(self, batch_size):
      
    self.rnn_hidden = torch.zeros((batch_size, self.rnn_hidden_dim)).to(self.device)

  def get_discount_reward(self, batch_reward):
    discount_rewards = []
    for reward in reversed(batch_reward):
        discounted_reward = 0
        for one_reward in reversed(reward):
            discounted_reward = one_reward + discounted_reward * self.gamma
            discount_rewards.insert(0, discounted_reward)
    return torch.Tensor(discount_rewards, device=self.device)

  def get_cov_mat(self):
    return self.cov_mat

  def save_model(self):
    torch.save(self.ppo_actor.state_dict(), f'./ppo_actors_weights.pt')
    torch.save(self.ppo_critic.state_dict(), f'./ppo_critic_weights.pt')

  def learn(self, batch, batch_size):

    log_probs = torch.cat(batch.log_prob)
    belief_map_batch = torch.cat(batch.belief_map)
    state_vector_batch = torch.cat(batch.state_vector)
    
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    discount_reward = self.get_discount_reward(reward_batch)

    self.init_hidden(batch_size)

    with torch.no_grad():
      state_values = self.ppo_critic(belief_map_batch, state_vector_batch).gather(1, action_batch)

      advantage_function = discount_reward - state_values

      advantage_function = ((advantage_function - advantage_function.mean()) / (advantage_function.std() + 1e-10)).unsqueeze(dim=-1)



      one_action_mean, self.rnn_hidden = self.ppo_actor(belief_map_batch, state_vector_batch, self.rnn_hidden)
      curr_state_values = self.ppo_critic(belief_map_batch, state_vector_batch)

      dist = MultivariateNormal(one_action_mean, self.cov_mat)
      curr_log_probs = dist.log_prob(action_batch)


      ratios = torch.exp(curr_log_probs - log_probs)
      surr1 = ratios * advantage_function
      surr2 = torch.clamp(ratios, 1 - self.train_config.ppo_loss_clip,
        1 + self.train_config.ppo_loss_clip) * advantage_function
      actor_loss = (-torch.min(surr1, surr2)).mean()

      self.optimizer_actor.zero_grad()
      actor_loss.backward()
      self.optimizer_actor.step()

      critic_loss = nn.MSELoss()(curr_state_values, discount_reward)
      self.optimizer_critic.zero_grad()
      critic_loss.backward()
      self.optimizer_critic.step()