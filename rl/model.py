# heavily based on https://github.com/vwxyzjn/cleanrl/
# referencing https://github.com/pranz24/pytorch-soft-actor-critic
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SoftQNetwork(nn.Module):
	def __init__(self, ob_space, ac_space):
		super().__init__()
		self.fc1 = nn.Linear(np.array(ob_space.shape).prod() + np.prod(ac_space.shape), 256) #env.single_observation_space.shape env.single_action_space.shape
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 1)

	def forward(self, x, a):
		x = torch.cat([x, a], 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5
epsilon = 1e-6


class Actor(nn.Module):
	def __init__(self, ob_space, ac_space):
		super().__init__()
		self.fc1 = nn.Linear(np.array(ob_space.shape).prod(), 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc_mean = nn.Linear(256, np.prod(ac_space.shape))
		self.fc_logstd = nn.Linear(256, np.prod(ac_space.shape))
		# action rescaling
		self.action_scale = torch.FloatTensor((ac_space.high - ac_space.low) / 2.0)
		self.action_bias = torch.FloatTensor((ac_space.high + ac_space.low) / 2.0)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		mean = self.fc_mean(x)
		log_std = self.fc_logstd(x)
		log_std = torch.tanh(log_std)
		log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

		return mean, log_std

	def get_action(self, x, training=True):
		mean, log_std = self(x)
		std = log_std.exp()
		normal = torch.distributions.Normal(mean, std)
		x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
		y_t = torch.tanh(x_t)
		action = y_t * self.action_scale + self.action_bias
		log_prob = normal.log_prob(x_t)
		# Enforcing Action Bound
		log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
		if training:
		  log_prob = log_prob.sum(1, keepdim=True)
		mean = torch.tanh(mean) * self.action_scale + self.action_bias
		return action, log_prob, mean

	def to(self, device):
		self.action_scale = self.action_scale.to(device)
		self.action_bias = self.action_bias.to(device)
		return super().to(device)

