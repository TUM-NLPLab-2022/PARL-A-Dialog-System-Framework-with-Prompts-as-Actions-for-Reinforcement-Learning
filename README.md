# PARL: A Dialog System Framework with Prompts as Actions for Reinforcement Learning


## 0 Poster

![](https://i.imgur.com/zVpqwAq.jpg)



## 1 File Structure
The repository structure is as follows:
- data/dailydialog: The daily dialog dataset which is used by the user chatbot to randomly sample an utterance in the beginning of the conversation.
- dyme_reward:  
  - config.py: File for defining the directory of models.
  - dyme_wrapper.pycompute metrics given an utterance.
  - environment.py: Class for environment of RL based on the Gym interface.
  - external_metrics_api.py
  - metric_helpers.py: Helper code to calculate metrics from Dyme original repository.
  - metrics.py: Definition of metric calculation. Adapted from Dyme original repository.
  - rewards.py: Definition of reward functions for the RL. 
  - models and [DYME
  - torchmoji: Directory with code for external model - TorchMoji (from [TorchMoji
- seq2seq_models: 
  - /blenderbot-400M-distill/fine-tuning.ipynb: codes for fine-tuning Blenderbot-400M-distil.
  - /blenderbot-400M-distill/example_run.ipynb: codes for example human interaction with the fine-tuned Blenderbot-400M-distill.
  - conversation.py: python class for conversation with the fine-tuned blenderbot, with augmented tokenizer and other specific processing.
- rl:
  - /RLmain.py: codes for training the RL agent.
  - /RLinfer.py: codes for multiple turns of interaction with the RL agent with fixed networks in the environment.
  - /RLinfer_single.py: codes providing an interface for using the RL agent with fixed networks for single turn of inference without the environment.
  - /model.py: codes of the definitions of RL models.
  - /sac.py: codes of SAC algorithm.
  - /rl.ipynb: example codes for installing and training the RL agent and doing interaction or inference with it.
- evaluation_and_results:
  - /blenderbot_responses.ipynb: codes for generating responses of the baseline for evaluation.
  - /RLPA_responses.ipynb: codes for generating responses of PARL for evaluation.
  - /generated_responses.csv: generated responses in summary.
  - /automatic_evaluation.ipynb: codes for automatic evaluation.
  - /automatic_evaluation_results.csv: results of automatic evaluation
  - /Human_evaluation_Krippendorff_s_Alpha_2.ipynb: codes for calculating Krippendorff's Alpha and adjacency matrices for inter-rater agreement in the human evaluation.
  - /Human_evaluation_randomizing_samples.ipynb: codes for randomizing samples for bline evaluation in the human evaluation.
  - /Human_evaluation_samples_to_be_evaluated_randomly_switched_2.csv: the randomized samples used in the human evaluation.

## 2 Installation <a name="sec2"></a>
Run the following scripts to install the requirements. Attention: due to the outdated libraries used by Conversational Sentence Encoder, you may be interested in using our frozen requirements with "--no-dependencies". 
```commandline
conda create -n dialogueGeneration python=3.7.13
conda activate dialogueGeneration
pip install -r requirements.txt --no-dependencies
```

Download models used for the reward calculation. Directory `dyme_models` has to be placed in the root of the project.\
The download link (valid until 08.09.2022): 
```
https://syncandshare.lrz.de/getlink/fi7H1aJhZwK9Zn2Qh3Gss9LT/dyme_models
```

## 3 Reproduce
Users can either run the codes on colab or locally. To run locally, make sure to first install dependencies as in [2 Installation](#sec2)
1. fine tune Blenderbot: run fine-tuning.ipynb
2. train policy network:
   ```commandline
    python rl/RLmain.py
   ```
    Attention: The python file will load saved models from "./savedmodels"; if no saved model exists, it will train one from scratch. For training from a saved checkpoint, please make sure you have the capacity to load all models and data including those huge ones used in DYME. If using Colab, Pro+ is recommended. \
    Defaut settings in arguments: 
    
    - total-timesteps: 1,000,000
    - batch-size: 256
    - learning-starts: 5e3
    - autosaving-per: 100
    - IO: False (set to True to enable interaction with human through IO) \

    All other arguments can be found in rl/RLmain.py

3. automatic evaluation
   - generate sample responses: run blenderbot_responses.ipynb and RLPA_responses.ipynb
   - run  automatic_evaluation.ipynb

## 4 Usage
  We have uploaded the following models and dataset on Huggingface:
  - [fine-tuned Blenderbot-400M-distil](https://huggingface.co/Adapting/dialogue_agent_nlplab2022)
  - [sentiment classifier for augmenting EmpatheticDialogues dataset](https://huggingface.co/Adapting/comfort_congratulations_neutral-classifier)
  - [augmented EmpatheticDialogues dataset](https://huggingface.co/datasets/Adapting/empathetic_dialogues_with_special_tokens)
  - [model weights of the policy network](https://huggingface.co/Adapting/PARL)
  
  For Inference/human interaction with PARL:\
  **Attention**: Make sure the trained model is in `./savedmodels`
  - Multiple turns of interaction with the fixed network in the environment:
     ```commandline
     python rl/RLinfer.py
     ```
  - Inference API without the environment:
    ```commandline
    python rl/RLinfer_single.py
    ```
    
## 5 Colab Example
[colab link](https://colab.research.google.com/gist/leoxiang66/3c6db947338d3ac887cef991fe5e1ee3/parl_example_1.ipynb)
> Note: It's recommended to have a RAM of 16GB for the running :)
  
