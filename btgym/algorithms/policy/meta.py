
from btgym.algorithms.utils import *

from btgym.research.casual_conv.policy import CasualConvPolicy_0_0


class MetaSubPolicy:
    """
    Stateful meta-optimisation policy to be hosted by of behavioral policy.
    Performs hyper-parameter and other type of meta-optimisation across different instances of hosting policy.
    """

    def __init__(self, task, num_host_policies, learn_rate, name='SubMetaPolicy'):
        """

        Args:
            task:                   int, host policy task, in [0, num_host_policies].
            num_host_policies:      total number of behavioral host policies instances to optimise across.
            learn_rate:             meta-policy learning rate
            name:                   name scope
        """
        with tf.variable_scope(name_or_scope=name):
            self.task = task
            self.learn_rate = learn_rate
            self.num_host_policies = num_host_policies

            self.input_stat_pl = tf.placeholder(dtype=tf.float32, name='in_stat_pl')

            self.input_stat = tf.reduce_mean(self.input_stat_pl)

            self.initial_cluster_value = tf.concat(
                [
                    tf.zeros(shape=[1, self.num_host_policies]),
                    tf.zeros(shape=[1, self.num_host_policies]),
                ],
                axis=0,
                name='initial_cluster_value'
            )

            self.cluster_averages_slot = tf.Variable(
                initial_value=self.initial_cluster_value,
                trainable=False,
                name='cluster_wide_averages_slot'
            )

            update_task_iteration = tf.scatter_nd_add(self.cluster_averages_slot, [[0, task]], [1])

            with tf.control_dependencies([update_task_iteration]):
                avg_prev = self.cluster_averages_slot[1, task]
                k = self.cluster_averages_slot[0, task]
                avg = avg_prev + (self.input_stat - avg_prev) / k
                self.update_op = tf.scatter_nd_update(self.cluster_averages_slot, [[1, task]], [avg])

            self.reset_op = tf.assign(
                self.cluster_averages_slot,
                self.initial_cluster_value
            )

            # Toy network:
            prob = tf.layers.dense(
                tf.expand_dims(self.cluster_averages_slot[1, :], axis=-1),
                units=10,
                activation=tf.nn.sigmoid,
                use_bias=False,
            )
            self.next_step_prob = tf.layers.dense(
                prob,
                units=1,
                activation=tf.nn.sigmoid,
                use_bias=False,
            )
            self.distribution = tf.distributions.Bernoulli(
                probs=tf.reduce_max(self.next_step_prob)
            )
            self.sample = self.distribution.sample()

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

            self.cluster_stat = tf.clip_by_value(
                # tf.reduce_mean(
                #     self.cluster_averages_slot
                # ),
                tf.expand_dims(self.cluster_averages_slot[1, :], axis=-1),
                -40,
                40
            )
            bound_avg = tf.sigmoid(- self.cluster_stat)
            self.loss = tf.reduce_mean(
                bound_avg * (1 - self.next_step_prob) + (1 - bound_avg) * self.next_step_prob
            )
            self.grads = tf.gradients(self.loss, self.var_list)

            self.summaries = [
                tf.summary.scalar('worker_avg_stat', self.cluster_averages_slot[1, task]),
                tf.summary.scalar('worker_iterations', self.cluster_averages_slot[0, task]),
                #tf.summary.histogram('clipped_cluster_stat', self.cluster_stat),
                tf.summary.scalar('loss', self.loss),
                tf.summary.histogram('next_step_prob', self.next_step_prob),
                tf.summary.scalar('grads_norm', tf.global_norm(self.grads))
            ]

    def update(self, input_stat):
        sess = tf.get_default_session()
        feed_dict = {self.input_stat_pl: input_stat}
        sess.run(self.update_op, feed_dict)

    def reset(self):
        sess = tf.get_default_session()
        sess.run(self.reset_op)

    def global_reset(self):
        raise NotImplementedError

    def act(self):
        """
        Do another step?

        Args:
            sess:
            iteration:

        Returns:

        """
        sess = tf.get_default_session()
        fetched = sess.run([self.sample])

        return fetched[-1]


class CasualMetaPolicy(CasualConvPolicy_0_0):

    def __init__(self, task=None,  cluster_spec=None, **kwargs):
        super(CasualMetaPolicy, self).__init__(**kwargs)
        if task is not None and cluster_spec is not None:
            self.meta = MetaSubPolicy(
                task=task,
                num_host_policies=len(cluster_spec['worker']),
                learn_rate=1e-3
            )

            self.var_list += self.meta.var_list



        else:
            print('Task ID or cluster_spec not provided, no meta-policy enabled.')






