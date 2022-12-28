# Code adapted from https://github.com/vwxyzjn/cleanrl/
# which references https://github.com/pranz24/pytorch-soft-actor-critic and https://github.com/openai/spinningup
# also referencing https://github.com/pranz24/pytorch-soft-actor-critic
import argparse
import os
from distutils.util import strtobool
import random
import sys
import time
import numpy as np
import torch
import gym
o_path = os.getcwd()
sys.path.append(o_path)
from dyme_reward import environment
from rl import sac
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import warnings
import copy
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
	help="the name of this experiment")
parser.add_argument("--seed", type=int, default=1,
	help="seed of the experiment")
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
	help="if toggled, `torch.backends.cudnn.deterministic=False`")
parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
	help="if toggled, cuda will be enabled by default")
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
	help="if toggled, this experiment will be tracked with Weights and Biases")
parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
	help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None,
	help="the entity (team) of wandb's project")
parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
	help="weather to capture videos of the agent performances (check out `videos` folder)")
# Algorithm specific arguments
parser.add_argument("--env-id", type=str, default="Chatbot",
	help="the id of the environment")
parser.add_argument("--total-timesteps", type=int, default=1000000,
	help="total timesteps of the experiments")
parser.add_argument("--buffer-size", type=int, default=int(1e6),
	help="the replay memory buffer size")
parser.add_argument("--gamma", type=float, default=0.99,
	help="the discount factor gamma")
parser.add_argument("--tau", type=float, default=0.005,
	help="target smoothing coefficient (default: 0.005)")
parser.add_argument("--batch-size", type=int, default=256,
	help="the batch size of sample from the reply memory")
parser.add_argument("--exploration-noise", type=float, default=0.1,
	help="the scale of exploration noise")
parser.add_argument("--learning-starts", type=int, default=5e3,
	help="timestep to start learning")
parser.add_argument("--policy-lr", type=float, default=3e-4,
	help="the learning rate of the policy network optimizer")
parser.add_argument("--q-lr", type=float, default=1e-3,
	help="the learning rate of the Q network network optimizer")
parser.add_argument("--policy-frequency", type=int, default=2,
	help="the frequency of training policy (delayed)")
parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
	help="the frequency of updates for the target nerworks")
parser.add_argument("--noise-clip", type=float, default=0.5,
	help="noise clip parameter of the Target Policy Smoothing Regularization")
parser.add_argument("--alpha", type=float, default=0.2,
	help="Entropy regularization coefficient.")
parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
	help="automatic tuning of the entropy coefficient")
parser.add_argument("--prompting", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
	help="enable prompting")
parser.add_argument("--autosaving-per", type=int, default=100,
	help="setting the epoch number for autosaving")
parser.add_argument("--IO", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
	help="enable input IO")
parser.add_argument("--showing-metrics", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
	help="enable metrics showing")
parser.add_argument("--showing-alpha", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
	help="enable alpha showing")
parser.add_argument("--inferring", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
	help="enable inferring")
args = parser.parse_args()

warnings.filterwarnings("ignore")
epoch_save_train_inf_for_colab = args.autosaving_per
save_model_path_colab = o_path+'/savedmodels'
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
	"hyperparameters",
	"|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic		

# Vectorized environments
def make_env(args):
    def thunk():
        seed = args.seed
        env = environment.EnvRL(args)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
envs = gym.vector.SyncVectorEnv([make_env(args)])
#envs = gym.vector.SyncVectorEnv([lambda: environment.EnvRL(args)] *3)

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# Agent
agent = sac.SAC(envs, args)
envs.single_observation_space.dtype = np.float32
rb = ReplayBuffer(
	args.buffer_size,
	envs.single_observation_space,
	envs.single_action_space,
	device,
	handle_timeout_termination=True,
)
start_time = time.time()

obs = envs.reset()
if os.path.exists(save_model_path_colab+'/savedmodel'+".t7"):
    save_path = save_model_path_colab+'/savedmodel'+".t7"
    checkpoint = torch.load(save_path)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.qf1.load_state_dict(checkpoint['qf1'])
    agent.qf2.load_state_dict(checkpoint['qf2'])
    agent.qf1_target.load_state_dict(checkpoint['qf1t'])
    agent.qf2_target.load_state_dict(checkpoint['qf2t'])
    agent.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    agent.alpha = copy.deepcopy(checkpoint["alpha"])
    rb = checkpoint['buffer']
    if args.autotune:
      agent.target_entropy = copy.deepcopy(checkpoint["target_entropy"])
      agent.log_alpha = copy.deepcopy(checkpoint["log_alpha"])
      agent.a_optimizer_reinitialize()
      agent.a_optimizer.load_state_dict(checkpoint["a_optimizer"])
    start_global_step = checkpoint['global_step']
    print("records: ", start_global_step)
    start_global_step += 1
else:
		start_global_step  = 0
for global_step in range(start_global_step, args.total_timesteps):
	print(global_step)
	if global_step < args.learning_starts: # random action at the beginning
		actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])  # Sample random action
	else:
		actions, _, _ = agent.actor.get_action(torch.Tensor(obs).to(device))  # Sample action from policy
		actions = actions.detach().cpu().numpy()
	#execute non-RL parts
	if args.prompting == False:
		for action in actions:
			action = np.array([0,0])
	next_states, rewards, dones, infos = envs.step(actions)

	for info in infos:
		if "episode" in info.keys():
			print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
			writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
			writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
			break
	real_next_obs = next_states.copy()
	#buffer
	for idx, d in enumerate(dones):
			if d:
				real_next_obs[idx] = infos[idx]["terminal_observation"]
	rb.add(obs, real_next_obs, actions, rewards, dones, infos)
	# ALGO LOGIC: training.
	obs = next_states
	if global_step > args.learning_starts:
		data = rb.sample(args.batch_size)
		qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss = agent.update_parameters(data)
		if global_step % args.policy_frequency == 0:
			for _ in range(args.policy_frequency):
				alpha, actor_loss, alpha_loss = agent.TD3delayed_update(data)
				if args.showing_alpha:
					print("updated alpha:",agent.alpha)
		if global_step % args.target_network_frequency == 0:
			agent.soft_update()
		if global_step % 100 == 0:
			writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
			writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
			writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
			writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
			writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
			writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
			writer.add_scalar("losses/alpha", alpha, global_step)
			print("SPS:", int(global_step / (time.time() - start_time)))
			writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
			if args.autotune:
				writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
			writer.flush()
		if global_step % epoch_save_train_inf_for_colab == 0:
			state = { 
					'actor': agent.actor.state_dict(),
          'qf1': agent.qf1.state_dict(),
          'qf2': agent.qf2.state_dict(),
          'qf1t': agent.qf1_target.state_dict(),
          'qf2t': agent.qf2_target.state_dict(),
          'q_optimizer':agent.q_optimizer.state_dict(), 
          'actor_optimizer':agent.actor_optimizer.state_dict(),
          'alpha':agent.alpha,
					'buffer':rb,
					'global_step':global_step
					}
			if args.autotune:
				state["target_entropy"] = agent.target_entropy
				state["log_alpha"] = agent.log_alpha
				state["a_optimizer"] = agent.a_optimizer.state_dict()
			torch.save(state,save_model_path_colab+'/savedmodel'+".t7")
			print('save train model at ', save_model_path_colab+'/savedmodel'+".t7","from colab")
envs.close()
writer.close()
