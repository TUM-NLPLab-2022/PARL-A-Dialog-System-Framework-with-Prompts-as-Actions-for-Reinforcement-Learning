# Code cut from environment.py and RLmain, which is adapted from https://github.com/vwxyzjn/cleanrl/
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
from rl import model as rlmodel
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import warnings
from conversational_sentence_encoder.vectorizers import SentenceEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from seq2seq_models import conversation as tconversation

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
	help="the name of this experiment")
parser.add_argument("--seed", type=int, default=1,
	help="seed of the experiment")
parser.add_argument("--env-id", type=str, default="Chatbot",
	help="the id of the environment")
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
	help="if toggled, `torch.backends.cudnn.deterministic=False`")
parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
	help="if toggled, cuda will be enabled by default")
parser.add_argument("--prompting", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
	help="enable prompting")
parser.add_argument("--inferring", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
	help="enable inferring")
args = parser.parse_args()

def input2output(uinput, sys_conv, encoder):
  chat_history = []
  chat_history.append(uinput)
  dialogue_encoded = encoder.encode_multicontext(chat_history)
  embedding = np.arctan(0.1 * dialogue_encoded) * 2 / np.pi
  obs = embedding.reshape(-1)
  action, _, _ = agent.get_action(torch.Tensor(obs).to(device), training=False)  # Sample action from policy
  action = action.detach().cpu().numpy()
  uinput = ' '.join(chat_history)
  prompt = behaviourandquestion(action)
  output = sys_conv.add_user_input(uinput, **prompt)
  return output

def behaviourandquestion(action):
    qu = action[1]
    be = action[0]
    prompt = dict()
    if qu > 0: #question
        prompt['qst'] = 'ask me for further details. '
        #behaviours
    if be < -1:
        prompt['bhv'] = "I'm in a negative mood, please comfort me."
    elif be >= -1 and be < 0:
        prompt['bhv'] = "give me some advice."
    elif be >= 1 and be < 2:
        prompt['bhv'] = "I'm in a positive mood, please congratulate me and praise me."
    return prompt

warnings.filterwarnings("ignore")
save_model_path_colab = o_path+'/savedmodels'
run_name = f"infer:{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Adapting/dialogue_agent_nlplab2022")
model = AutoModelForSeq2SeqLM.from_pretrained("Adapting/dialogue_agent_nlplab2022", revision = 'b86f62986872b4c1a9921acdb8cd226761d736cf')
sys_conv = tconversation.Conversation(model, tokenizer, 128, device)
multicontext_encoder = SentenceEncoder(multiple_contexts=True)
# Agent
actsp = gym.spaces.Box(np.array([-2,-1]), np.array([+2,+1])) 
obsp = gym.spaces.Box(np.array([-1]*512), np.array([1]*512))
actsp.seed(seed)
obsp.seed(seed)
agent = rlmodel.Actor(obsp, actsp)

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

print("User:")
uinput = input()
output = input2output(uinput, sys_conv)
print(f'''>> User:{uinput}''',flush=True)
print(f'''>> Bot: {output}''',flush=True)
writer.close()