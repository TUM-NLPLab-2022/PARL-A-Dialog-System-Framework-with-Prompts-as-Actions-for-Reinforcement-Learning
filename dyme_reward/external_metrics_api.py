import numpy as np
import torch
import json
from models.empathy_classifier import EmpathyClassifier
from models.infersent import InferSent
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis, torchmoji_feature_encoding
from config import models_dir
import nltk

nltk.download('punkt')


class EmpathyClassifierAPI:

    def __init__(self, device):
        self.classifier = EmpathyClassifier(device=device)

    def compute_empathy_levels(self, utterance_pairs):
        empathy_levels = []
        for pair in utterance_pairs:
            (_, predictions_ER, _, predictions_IP, _, predictions_EX, _, _, _, _, _, _) = \
                self.classifier.predict_empathy([pair[0]], [pair[1]])
            empathy_levels.append([predictions_ER[0], predictions_IP[0], predictions_EX[0]])
        empathy_levels = np.array(empathy_levels)
        return empathy_levels


class InferSentAPI:
    model_path = models_dir.joinpath('inferSent/encoder/infersent1.pickle')
    w2v_path = models_dir.joinpath('inferSent/dataset/GloVe/glove.6B.300d.txt')
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}

    def __init__(self):
        self.model = InferSent(self.params_model)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.set_w2v_path(self.w2v_path)
        self.model.build_vocab_k_words(K=100000)
        self.model.eval()

    def encode_multiple(self, utterances):
        filtered_utterances = ['_' if u == '' else u for u in utterances]
        return self.model.encode(filtered_utterances, tokenize=True)


class DeepmojiAPI:
    vocab_path = models_dir.joinpath('torchMoji/vocabulary.json')
    model_path = models_dir.joinpath('torchMoji/pytorch_model.bin')

    def __init__(self):
        with open(self.vocab_path, 'r') as f:
            vocabulary = json.load(f)
        maxlen = 30
        self.tokenizer = SentenceTokenizer(vocabulary, maxlen)
        self.model = torchmoji_emojis(self.model_path)
        self.model.eval()

    def encode_multiple(self, utterances):
        filtered_utterances = ['_' if u == '' else u for u in utterances]
        tokenized, _, _ = self.tokenizer.tokenize_sentences(filtered_utterances)
        return self.model(tokenized)
