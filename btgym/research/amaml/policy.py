import tensorflow as tf
from btgym.research import AacStackedRL2Policy


class AacMetaCriticPolicy(AacStackedRL2Policy):

    def __init__(self, **kwargs):
        super(AacMetaCriticPolicy, self).__init__(**kwargs)

        self.meta_critic_var_list =\
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*lstm_2.*') +\
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tf.get_variable_scope().name + '.*aac_dense_vfn.*')

        self.actor_var_list = [var for var in self.var_list if var not in self.meta_critic_var_list]