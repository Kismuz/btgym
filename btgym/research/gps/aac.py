import tensorflow as tf

from btgym.algorithms import BaseAAC
from .loss import guided_aac_loss_def_0_0


class GuidedAAC(BaseAAC):
    """
    Actor-critic framework extended with expert actions imitation loss.
    """
    def __init__(
            self,
            expert_loss=guided_aac_loss_def_0_0,
            aac_lambda=1.0,
            guided_lambda=1.0,
            _log_name='GuidedA3C',
            **kwargs
    ):
        """

        Args:
            expert_loss:        callable returning tensor holding on_policy imitation loss graph and summaries
            aac_lambda:         float, main on_policy a3c loss lambda
            guided_lambda:      float, imitation loss lambda
            _log_name:          str, class-wide logger name, internal do not use
            **kwargs:           see BaseAAC kwargs
        """
        try:
            super(GuidedAAC, self).__init__(_log_name=_log_name, **kwargs)
            with tf.device(self.worker_device):
                with tf.variable_scope('local'):
                    guided_loss_ext, guided_summary_ext = expert_loss(
                        pi_actions=self.local_network.on_logits,
                        expert_actions=self.local_network.expert_actions,
                        name='on_policy',
                        verbose=True
                    )
                    self.loss = aac_lambda * self.loss + guided_lambda * guided_loss_ext
                    extended_summary = [guided_summary_ext, tf.summary.scalar("recalc_total_loss", self.loss)]

                    self.log.notice('aac_lambda: {}, guided_lambda: {}'.format(aac_lambda, guided_lambda))

                    # Override train op def:
                    self.grads, _ = tf.clip_by_global_norm(
                        tf.gradients(self.loss, self.local_network.var_list),
                        40.0
                    )
                    grads_and_vars = list(zip(self.grads, self.network.var_list))
                    self.train_op = self.optimizer.apply_gradients(grads_and_vars)

                    # Merge summary:
                    extended_summary.append(self.model_summary_op)
                    self.model_summary_op = tf.summary.merge(extended_summary, name='extended_summary')

        except:
            msg = 'Child 0.0 class __init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)
