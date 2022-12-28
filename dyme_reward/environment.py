import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from dyme_wrapper import DymeWrapper
from rewards import mse_reward, weighted_mse_reward, weighted_rmse_reward, vector_difference_reward
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from transformers import ConversationalPipeline, Conversation, pipeline
from conversational_sentence_encoder.vectorizers import SentenceEncoder
from seq2seq_models import conversation as tconversation
from conversational_sentence_encoder.vectorizers import SentenceEncoder
import torch
from gym import spaces, core
import numpy as np
import warnings

daily_dialog_path = './data/dailydialog/train/dialogues_train.txt' #..
tokenizer = AutoTokenizer.from_pretrained("Adapting/dialogue_agent_nlplab2022")
model = AutoModelForSeq2SeqLM.from_pretrained("Adapting/dialogue_agent_nlplab2022", revision = 'b86f62986872b4c1a9921acdb8cd226761d736cf')

MAX_DIALOG_LENGTH = 10

REWARD_FUNCTION = {0: mse_reward,
                   1: weighted_mse_reward,
                   2: weighted_rmse_reward,
                   3: vector_difference_reward,
                   }

warnings.filterwarnings("ignore")
class EnvRL(core.Env):
    def __init__(self, args, model_checkpoint = "facebook/blenderbot-1B-distill", reward_func=0, tokenizer=tokenizer, model=model):
        """
        model_checkpoint: The model checkpoint of the user chatbot
        reward_func: 0 for mse_reward, 1 for weighted_mse_reward, 2 for weighted_rmse_reward and
                     3 for vector_difference_reward
        tokenizer: Tokenizer of the backbone dialogue system
        model: Seq2seq model of the backbone dialogue system
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.model_max_length = 128
        self.args = args

        # Choice of reward function
        self.reward_func = reward_func

        if args.IO == False:
          # Initialize user chatbot
          self.usr_chatbot = pipeline("conversational", model = model_checkpoint)

          # user chatbot: conversation object
          self.usr_conv = Conversation()

        # conversation object
        self.sys_conv = tconversation.Conversation(model, tokenizer, self.model_max_length, self.device)

        # Initialize chat history
        self.chat_history = []

        # Initialize Dyme Wrapper
        self.dyme_wrapper = DymeWrapper()

        # Initialize Seq2seq system chatbot
        #self.sys_chatbot = ConversationalPipeline(model=model, tokenizer=tokenizer)

        # dialogue_encoded 
        self.multicontext_encoder = SentenceEncoder(multiple_contexts=True)
        # Bert
        self.bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        self.bert_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

        # Spaces
        self.action_space = spaces.Box(np.array([-2,-1]), np.array([+2,+1]))  #behaviour, question or not
        self.observation_space =  spaces.Box(np.array([-1]*512), np.array([1]*512)) #Bert:768

    def step(self, action):
        """
        This takes a step in the environment by performing an action. This returns the next observation, reward,
        a done flag, and info. If the done flag == True, then the chat history is refreshed.
        
        action: A (2,) array where the first dimension represents behavior actions and the second represents question-or-not actions.
        """
        # Get input
        input = ' '.join(self.chat_history)
        #if self.args.inferring == False:
        #  print("Input: ", input)

        # Turn action into behaviours and (if any) question
        prompt = self.behaviourandquestion(action)

        # Concatenate action with the history and get the output of the system chatbot
        output = self._get_system_response(input, prompt)
        if self.args.inferring == False:
          print("Output: ", output)

        # Calculate reward
        if self.args.inferring == False:
          prediction = self.dyme_wrapper.predict_metrics(self.chat_history)
          metrics_for_response = self.dyme_wrapper.compute_metrics_for_response(self.chat_history, output)
          reward = REWARD_FUNCTION[self.reward_func](metrics_for_response, prediction)
          if self.args.showing_metrics == True:
            print("Metrics: ", metrics_for_response)
            print("Prediction: ", prediction) 
            print("Reward:", reward)
        else:
          reward = 0

        # Add to chat history
        self._add_to_chat_history(output)  
        # Check if done for no-human
        done = False
        if self.args.IO == False:
          done = len(self.chat_history) == MAX_DIALOG_LENGTH
        
        # Get observation
        uinput = self._next_userinput()
        if uinput == 'DONE':
          done = True
          obs = self.next_observation('')
        else:
          obs = self.next_observation(uinput)
        # providing info
        info = {"terminal_observation": False, "action": action, "prompt": prompt, "response": output, "user_input": uinput}
        if done:
            info["terminal_observation"] = True
        return obs, reward, done, info

    def get_output_infer(self, action):
        """
        This takes a step in the environment with an action and a user input. This returns the next observation, reward,
        a done flag, and info. If the done flag == True, then the chat history is refreshed.
        """
        # Get input
        input = ' '.join(self.chat_history)
        #print("Input: ", input)

        # Turn action into behaviours and (if any) question
        prompt = self.behaviourandquestion(action)

        # Concatenate action with the history and get the output of the system chatbot
        output = self._get_system_response(input, prompt)
        reward = 0
        # Add to chat history
        self._add_to_chat_history(output)
        info_stage1 = {"action": action, "prompt": prompt, "response": output}
        return output, info_stage1

    def get_new_input(self, uinput, done=None):
        # Get observation
        obs = self.next_observation(uinput)
        # providing info
        info_stage2 = {"terminal_observation": False, "user_input": uinput}
        if done:
            info_stage2["terminal_observation"] = True
        return obs, info_stage2

    def reset(self):
        """Reset the state of the environment to an initial state"""
        self.chat_history.clear()
        if self.args.IO == False:
          self.usr_conv = Conversation()
        self.sys_conv = tconversation.Conversation(model, tokenizer, self.model_max_length, self.device)
        uinput = self._next_userinput()
        obs = self.next_observation(uinput)
        return obs

    def reset_infer(self, uinput):
        """Reset the state of the environment to an initial state with a userinput"""
        self.chat_history.clear()
        if self.args.IO == False:
          self.usr_conv = Conversation()
        self.sys_conv = tconversation.Conversation(model, tokenizer, self.model_max_length, self.device)
        done = False
        if uinput == 'DONE':
          done = True
          uinput = self._next_userinput()
        obs = self.next_observation(uinput)
        info = {"terminal_observation": False, "user_input": uinput}
        if done:
            info["terminal_observation"] = True
        return obs, info
    
    def _get_system_response(self, new_usr_input, prompt):
        # Get qst and bhv from the prompt, adding a new line into the history, get response
        resp = self.sys_conv.add_user_input(new_usr_input, **prompt)
        
        # Add the response to the user pipeline
        if self.args.IO == False:
          self.usr_conv.add_user_input(resp)
        
        return resp
    def _next_userinput(self):
        if self.args.IO == False:
          if len(self.chat_history):
              self.usr_chatbot(self.usr_conv)
              resp = self.usr_conv.generated_responses[-1]
          else:
              convs = _get_dataset(daily_dialog_path)
              n = random.randint(0, len(convs))
              resp = convs[n][0]
          print("Respond:", resp)
        else:
          print("Please enter:")
          resp = input()
        return resp

    def next_observation(self, uinput):
        self._add_to_chat_history(uinput)
        # Get BERT embedding
        dialogue_encoded = self.multicontext_encoder.encode_multicontext(self.chat_history)
        #encoded_input = self.bert_tokenizer(' '.join(self.chat_history), padding=True, return_tensors='pt')
        #with torch.no_grad():
        #    model_output = self.bert_model(**encoded_input)
        #embedding = _mean_pooling(model_output, encoded_input['attention_mask'])
        #normalisation to (-1, 1)
        embedding = np.arctan(0.1 * dialogue_encoded) * 2 / np.pi
        return embedding.reshape(-1)

    def _add_to_chat_history(self, utterance):
        self.chat_history.append(utterance)
    #convert sampled action to actual prompts
    
    def behaviourandquestion(self, action):
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
        if self.args.showing_metrics == True:
          print(prompt)
        return prompt        
        
def _get_dataset(file_name):
    conversations = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split("__eou__")
            conversations.append(values)
    return conversations


#Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
