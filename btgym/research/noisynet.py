from btgym.algorithms import BaseAAC
from btgym.algorithms.nn.layers import noisy_linear


class NoisyNetUnreal(BaseAAC):
    """
    Noisy Net Unreal implementation.
    """

    def __init__(self, **kwargs):
        kwargs['_log_name'] = 'NoisyNetUnreal'
        #kwargs['model_beta'] = 0.0
        kwargs['policy_config']['linear_layer_ref'] = noisy_linear
        super(NoisyNetUnreal, self).__init__(**kwargs)