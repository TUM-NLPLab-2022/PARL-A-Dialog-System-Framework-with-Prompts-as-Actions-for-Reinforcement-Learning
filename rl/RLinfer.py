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
from rl import model
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import warnings

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
parser.add_argument("--IO", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
	help="enable input IO")
parser.add_argument("--showing-metrics", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
	help="enable metrics showing")
parser.add_argument("--showing-alpha", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
	help="enable alpha showing")
parser.add_argument("--inferring", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
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
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = args.torch_deterministic		

env = environment.EnvRL(args)
env = gym.wrappers.RecordEpisodeStatistics(env)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
# Agent
agent = model.Actor(env.observation_space, env.action_space)
start_global_step = 0
if os.path.exists(save_model_path_colab+'/savedmodel'+".t7"):
    save_path = save_model_path_colab+'/savedmodel'+".t7"
    checkpoint = torch.load(save_path)
    agent.load_state_dict(checkpoint['actor'])
    start_global_step = checkpoint['global_step']
    print("records: ", start_global_step)
    start_global_step += 1
else:
		print("Alarm. No existing model.")
agent = agent.to(device)
agent.eval()
global_step = start_global_step

print("User beginning:")
uinput = input()
obs, info = env.reset_infer(uinput)
while uinput != 'EXIT':
	print(f'''>> User:{uinput}''',flush=True)
	action, _, _ = agent.get_action(torch.Tensor(obs).to(device), training=False)  # Sample action from policy
	action = action.detach().cpu().numpy()
	output, _ = env.get_output_infer(action)
	print(f'''>> Bot: {output}''',flush=True)
	if "episode" in info.keys():
		print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
		writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
		writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
	#execute non-RL parts
	print("User:")
	uinput = input()
	done = False
	if uinput == 'DONE':
		done = True
		print("User:")
		uinput = input()
		obs, info = env.reset_infer(uinput)
	else:
		obs, info = env.get_new_input(uinput, done)
	global_step += 1
env.close()
writer.close()