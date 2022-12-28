#Code from https://github.com/florianvonunold/DYME

import torch
from torch import nn
from transformers import BertModel


class DYME(nn.Module):
    """
    BERT (used as context encoder) with a regression head.
    For details refer to the paper linked in the readme.
    """
    def __init__(self, num_metrics=1, max_prediction_position=16, include_numerical_features=True, tod_bert=False):
        print('Initializing DYME ...')
        super(DYME, self).__init__()
        self.model_type = 'bert'
        self.num_metrics = num_metrics
        self.include_numerical_features = include_numerical_features

        # initialize encoder
        if tod_bert:
            self.bert = BertModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        # freeze bert
        for p in self.bert.parameters():
            p.requires_grad = False

        # calculate hidden size
        self.hidden_size = 768 + (self.num_metrics * max_prediction_position + 1) * include_numerical_features
        # bert embedding size + (number of metrics * number of utterances until max_prediction_position) + 1
        # (the +1 is for the scalar 'prediction position')

        # init additional layers (i.e. the classifier head)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, num_metrics)
        print('Successfully initialized DYME!')

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, numerical_features=None, prediction_positions=None):
        output_dict = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = output_dict['pooler_output']  # shape (batch_size, 768)

        # apply dropout to the sequence output
        pooled_output_dropout = self.dropout(pooled_output)  # shape (batch_size, 768)

        # concatenate BERT embedding with feature vector
        if self.include_numerical_features:
            full_combined_feature_vector = torch.cat((pooled_output_dropout, numerical_features), dim=1)
        else:
            full_combined_feature_vector = pooled_output_dropout

        # classification head
        x = self.dense(full_combined_feature_vector)
        x = torch.relu(x)
        x = self.dropout(x)

        # linear output projection
        logits = self.out(x)

        return logits
