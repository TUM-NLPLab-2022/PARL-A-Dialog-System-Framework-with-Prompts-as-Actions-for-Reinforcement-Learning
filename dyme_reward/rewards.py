import numpy as np
from sklearn.metrics import mean_squared_error


def mse_reward(computed_metrics, prediction, weights=None):
    weights = _check_weights(computed_metrics, weights)
    difference = (computed_metrics - prediction)**2
    error = np.sum(difference * weights) / len(computed_metrics)
    return -error


def weighted_mse_reward(computed_metrics, prediction, weights=None):
    weights = _check_weights(computed_metrics, weights)
    return -mean_squared_error(computed_metrics, prediction, sample_weight=weights)


def weighted_rmse_reward(computed_metrics, prediction, weights=None):
    weights = _check_weights(computed_metrics, weights)
    return -mean_squared_error(computed_metrics, prediction, sample_weight=weights, squared=False)


def vector_difference_reward(computed_metrics, prediction, weights=None):
    weights = _check_weights(computed_metrics, weights)
    return -np.absolute(computed_metrics - prediction) * weights


def _check_weights(computed_metrics, weights):
    if weights is None:
        return np.ones_like(computed_metrics)
    else:
        m_shape = computed_metrics.shape
        assert weights.shape == m_shape, f'weights have to be of shape: {m_shape}'
