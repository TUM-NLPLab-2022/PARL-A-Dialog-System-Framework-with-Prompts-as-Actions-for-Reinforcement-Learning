import numpy as np
import pickle
import torch
from transformers import BertTokenizer
import spacy
from models.dyme import DYME
import metrics
from config import models_dir

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

dyme_path = models_dir.joinpath('dyme/dailydialog_empatheticdialogues_2_8_default_metrics')
scalers_path = models_dir.joinpath('dyme/dailydialog_empatheticdialogues_2_8_default_metrics_scalers.pickle')
dialog_lengths = list(range(2, 9))
max_prediction_position = dialog_lengths[-1] - 1
use_numerical_features = True
use_tod_bert = False
metric_names = ['question',
                'conversation_repetition',
                'self_repetition',
                'utterance_repetition',
                'word_repetition',
                'utterance_length',
                'infersent_coherence',
                'USE_similarity',
                'word2vec_coherence',
                'deepmoji',
                'empathy']
all_metrics = ['question',
               'conversation_repetition',
               'self_repetition',
               'utterance_repetition',
               'word_repetition',
               'utterance_length',
               'infersent_coherence',
               'USE_similarity',
               'word2vec_coherence',
               'deepmoji_sentiment',
               'deepmoji_coherence',
               'emotional_reaction_level',
               'interpretation_level',
               'exploration_level']


def _load_dyme():
    model = DYME(num_metrics=len(all_metrics),
                 max_prediction_position=max_prediction_position,
                 include_numerical_features=use_numerical_features,
                 tod_bert=use_tod_bert)
    model.load_state_dict(torch.load(dyme_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _load_scalers():
    with open(scalers_path, 'rb') as scalers:
        scalers = pickle.load(scalers)
        feature_scaler = scalers['feature_scaler']
        label_scaler = scalers['label_scaler']
    return feature_scaler, label_scaler


def _compute_metrics_for_conversation(conversation, metric_names, judging):
    """
    Computes the specified metrics for a given conversation
    Code from https://github.com/florianvonunold/DYME
    """
    """
      Parameters:
      conversation : list
          A list of utterances, every utterance is also a list for the words and puncts in it
      judging: int
          indicating if this fuction is used for the system output or by DYME

      Returns:
      conversation_metrics : np.array
          metrics for sentences respectively
    """
    conversation_metrics = []

    for metric in metric_names:
        metric_calculator = getattr(metrics, metric)  # get metric calculation function
        # check if it's the openning; empathy metrics are sensitive to this
        if len(conversation) == 1 and metric == 'empathy': 
          # set the shape according whether it's for the system output: (2,3) for output and (1,3) for DYME
          involvedsentence = judging + 1  
          cur_metric_for_conv = np.zeros(shape=(involvedsentence, 3)).astype(int)
        else:  
          cur_metric_for_conv = metric_calculator(conversation)  # apply metric calcuation function
        if metric == 'deepmoji':  # save computation time by not computing the deepmoji embeddings twice
            conversation_metrics.append(cur_metric_for_conv[0])  # deepmoji sentiment
            conversation_metrics.append(cur_metric_for_conv[1])  # deepmoji coherence
        elif metric == 'empathy':
            conversation_metrics.append(cur_metric_for_conv[:, 0])  # emotional_reaction_level in the conversation
            conversation_metrics.append(cur_metric_for_conv[:, 1])  # interpretation_level in the conversation
            conversation_metrics.append(cur_metric_for_conv[:, 2])  # exploration_level in the conversation
        else:
            conversation_metrics.append(cur_metric_for_conv)
    return np.array(conversation_metrics)


class DymeWrapper:
    def __init__(self):
        self.dyme = _load_dyme()
        self.feature_scaler, self.label_scaler = _load_scalers()
        if use_tod_bert:  # TOD-BERT tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('TODBERT/TOD-BERT-JNT-V1')
        else:  # BERT tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.nlp = spacy.load("en_core_web_sm")

    def compute_metrics_for_response(self, chat_history, response):
        """
        Parameters:
        chat_history : list
            A list of utterances in one dialog
        response: str
            Response created by the model

        Returns:
        scaled_metrics : array
            Normalized values of all metrics for the response
        """
        dialog = chat_history.copy()
        dialog.append(response)
        utterances = [[token.text for token in self.nlp(utterance.lower()) if token.text != ' '] for utterance in dialog]
        metrics_all = _compute_metrics_for_conversation(utterances, metric_names, 1)

        metrics_for_response = metrics_all[:, -1]
        scaled_metrics = self.label_scaler.transform(metrics_for_response.reshape(1, -1))
        scaled_metrics = scaled_metrics[0]
        return scaled_metrics

    def predict_metrics(self, chat_history):
        """
        Parameters:
        chat_history : list
            A list of utterances in one dialog, only last max_prediction_position utterances are used for prediction

        Returns:
        prediction : array
            Normalized predicted values for all metrics
        """
        dialog = chat_history.copy()
        dialog = dialog[-max_prediction_position:]
        dialog = [utterance.lower() for utterance in dialog]
        dialog_text = [' '.join(dialog)]

        encoding = self.tokenizer(dialog_text, truncation=True, padding=True)
        features = self.__compute_features(dialog, metric_names)

        logits = self.dyme(input_ids=torch.tensor(encoding['input_ids'], dtype=torch.int).to(device),
                           attention_mask=torch.tensor(encoding['attention_mask'], dtype=torch.int).to(device),
                           token_type_ids=torch.tensor(encoding['token_type_ids'], dtype=torch.int).to(device),
                           numerical_features=torch.tensor(features, dtype=torch.float).to(device))

        prediction = logits[0].cpu().detach().numpy()
        return prediction

    def __compute_features(self, dialog, metric_names):
        utterances = [[token.text for token in self.nlp(utterance) if token.text != ' '] for utterance in dialog]
        metrics = _compute_metrics_for_conversation(utterances, metric_names, 0)
        predict_at_position = len(dialog)
        numerical_features = self.__prepare_features(metrics, predict_at_position, max_prediction_position)
        return numerical_features

    def __prepare_features(self, metrics, predict_at_position, max_prediction_position):
        imputation = np.full((metrics.shape[0], max_prediction_position - predict_at_position), np.NaN)
        features = np.concatenate((metrics, imputation), axis=1)
        features = features.reshape(-1, features.shape[0] * features.shape[1], order='F')
        features = np.c_[features, np.array(predict_at_position)]
        features_scaled = self.feature_scaler.transform(features)
        features_scaled[np.isnan(features_scaled)] = -1
        return features_scaled
