# Code from https://github.com/behavioral-data/Empathy-Mental-Health

import torch
from torch import nn


class Baseline(nn.Module):
    """
    Uses the input mean per metric as prediction (feature mean per metric).
    :param baseline_type: choose between 'mean' (default) and 'last'
        'mean': uses the per metric average of all input utterances (default)
        'last': uses the metric values of the last input utterance
    """
    def __init__(self, num_metrics=1, baseline_type='mean'):
        print('Initializing Baseline!')
        super(Baseline, self).__init__()
        self.num_metrics = num_metrics
        self.model_type = 'baseline'
        self.baseline_type = baseline_type
        print('Successfully initialized DYME!')

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, numerical_features=None, prediction_positions=None):
        # last indices of metric features
        last_indices_of_metric_features = (prediction_positions * self.num_metrics).to(torch.int64)

        if self.baseline_type == 'last':
            first_indices_of_metric_features = ((prediction_positions - 1) * self.num_metrics).to(torch.int64)
            logits = [sample_features[first_indices_of_metric_features[idx]:last_indices_of_metric_features[idx]]
                      for idx, sample_features in enumerate(numerical_features)]

        else:  # self.baseline_type == 'mean' (default)
            # for every sample in the batch (numerical_features),
            # get all metric features for all input utterances and average them per metric
            # average per metric
            logits = [torch.mean(sample_features[:last_indices_of_metric_features[idx]].view(-1, self.num_metrics), dim=0)
                      for idx, sample_features in enumerate(numerical_features)]

        return torch.stack(logits)  # list of tensors to 2d tensor
