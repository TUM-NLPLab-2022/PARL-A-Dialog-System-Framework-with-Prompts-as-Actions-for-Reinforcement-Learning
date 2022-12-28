# heavily based on https://github.com/vwxyzjn/cleanrl/
# referencing https://github.com/pranz24/pytorch-soft-actor-critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import SoftQNetwork, Actor

class SAC(object):
	def __init__(self, envs, args):
		self.args = args
		self.alpha = args.alpha
		self.gamma = args.gamma
		self.device = torch.device("cuda" if self.args.cuda else "cpu")
		self.max_action = float(envs.single_action_space.high[0])
		self.actor = Actor(envs.single_observation_space, envs.single_action_space).to(self.device)
		self.qf1 = SoftQNetwork(envs.single_observation_space, envs.single_action_space).to(self.device)
		self.qf2 = SoftQNetwork(envs.single_observation_space, envs.single_action_space).to(self.device)
		self.qf1_target = SoftQNetwork(envs.single_observation_space, envs.single_action_space).to(self.device)
		self.qf2_target = SoftQNetwork(envs.single_observation_space, envs.single_action_space).to(self.device)
		self.qf1_target.load_state_dict(self.qf1.state_dict())
		self.qf2_target.load_state_dict(self.qf2.state_dict())
		self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr)
		self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args.policy_lr)

		# Automatic entropy tuning
		if self.args.autotune:
			self.target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(self.device)).item()
			self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha = self.log_alpha.exp().item()
			self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
		else:
			self.alpha = self.args.alpha

	def update_parameters(self, data):
		#training-updating
		with torch.no_grad():
			next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
			qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
			qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
			min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
			next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

		qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
		qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
		qf1_loss = F.mse_loss(qf1_a_values, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
		qf2_loss = F.mse_loss(qf2_a_values, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
		qf_loss = qf1_loss + qf2_loss

		self.q_optimizer.zero_grad()
		qf_loss.backward()
		self.q_optimizer.step()

		return qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss

	def TD3delayed_update(self, data):
		pi, log_pi, _ = self.actor.get_action(data.observations)
		qf1_pi = self.qf1(data.observations, pi)
		qf2_pi = self.qf2(data.observations, pi)
		min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
		actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()
		if self.args.autotune:
			with torch.no_grad():
				_, log_pi, _ = self.actor.get_action(data.observations)
			alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()
			self.a_optimizer.zero_grad()
			alpha_loss.backward()
			self.a_optimizer.step()
			alpha = self.log_alpha.exp().item()
			self.alpha = alpha
		else: 
			alpha_loss = 0
		return self.alpha, actor_loss, alpha_loss

	def soft_update(self):
		for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
			target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
		for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
			target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

	def a_optimizer_reinitialize(self):
		self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
