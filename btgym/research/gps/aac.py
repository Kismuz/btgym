import tensorflow as tf

from btgym.algorithms import BaseAAC
from .loss import guided_aac_loss_def_0_0, guided_aac_loss_def_0_1, guided_aac_loss_def_0_3
from btgym.algorithms.runner.synchro import BaseSynchroRunner, VerboseSynchroRunner


class GuidedAAC(BaseAAC):
    """
    Actor-critic framework augmented with expert actions imitation loss:
    L_gps = aac_lambda * L_a3c + guided_lambda * L_im.

    This implementation is loosely refereed as 'guided policy search' after algorithm described in paper
    by S. Levine and P. Abbeel `Learning Neural Network Policies with Guided PolicySearch under Unknown Dynamics`

    in a sense that exploits idea of fitting 'local' (here - single episode) oracle for environment with
    generally unknown dynamics and use actions demonstrated by it to optimize trajectory distribution for training agent.

    Note that this particular implementation of expert does not provides
    complete action-state space trajectory for agent to follow.
    Instead it estimates `advised` categorical distribution over actions conditioned on `external` (i.e. price dynamics)
    state observations only.

    Papers:
        - Levine et al., 'Learning Neural Network Policies with Guided PolicySearch under Unknown Dynamics'
            https://people.eecs.berkeley.edu/~svlevine/papers/mfcgps.pdf

        - Brys et al., 'Reinforcement Learning from Demonstration through Shaping'
            https://www.ijcai.org/Proceedings/15/Papers/472.pdf

        - Wiewiora et al., 'Principled Methods for Advising Reinforcement Learning Agents'
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.6412&rep=rep1&type=pdf

    """
    def __init__(
            self,
            expert_loss=guided_aac_loss_def_0_3,
            aac_lambda=1.0,
            guided_lambda=1.0,
            guided_decay_steps=None,
            runner_config=None,
            # aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            aux_render_modes=None,
            name='GuidedA3C',
            **kwargs
    ):
        """

        Args:
            expert_loss:        callable returning tensor holding on_policy imitation loss graph and summaries
            aac_lambda:         float, main on_policy a3c loss lambda
            guided_lambda:      float, imitation loss lambda
            guided_decay_steps: number of steps guided_lambda is annealed to zero
            name:               str, name scope
            **kwargs:           see BaseAAC kwargs
        """
        try:
            self.expert_loss = expert_loss
            self.aac_lambda = aac_lambda
            self.guided_lambda = guided_lambda * 1.0
            self.guided_decay_steps = guided_decay_steps
            self.guided_lambda_decayed = None
            self.train_guided_lambda = None
            if runner_config is None:
                runner_config = {
                    'class_ref': BaseSynchroRunner,
                    'kwargs': {
                        'aux_render_modes': aux_render_modes,  # ('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
                    }
                }

            super(GuidedAAC, self).__init__(
                runner_config=runner_config,
                name=name,
                aux_render_modes=aux_render_modes,
                **kwargs
            )
        except:
            msg = 'GuidedAAC.__init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def _make_loss(self, **kwargs):
        """
        Augments base loss with expert actions imitation loss

        Returns:
            tensor holding estimated loss graph
            list of related summaries
        """
        aac_loss, summaries = self._make_base_loss(**kwargs)

        # Guidance annealing:
        if self.guided_decay_steps is not None:
            self.guided_lambda_decayed = tf.train.polynomial_decay(
                self.guided_lambda,
                self.global_step + 1,
                self.guided_decay_steps,
                0.0,
                power=1,
                cycle=False,
            )
        else:
            self.guided_lambda_decayed = self.guided_lambda
        # Switch to zero when testing - prevents information leakage:
        self.train_guided_lambda = self.guided_lambda_decayed * tf.cast(self.local_network.train_phase, tf.float32)

        self.guided_loss, guided_summary = self.expert_loss(
            pi_actions=self.local_network.on_logits,
            expert_actions=self.local_network.expert_actions,
            name='on_policy',
            verbose=True,
            guided_lambda=self.train_guided_lambda
        )
        loss = self.aac_lambda * aac_loss + self.guided_loss

        summaries += guided_summary

        self.log.notice(
            'guided_lambda: {:1.6f}, guided_decay_steps: {}'.format(self.guided_lambda, self.guided_decay_steps)
        )

        return loss, summaries


class VerboseGuidedAAC(GuidedAAC):
    """
    Extends parent `GuidedAAC` class with additional summaries.
    """

    def __init__(
            self,
            runner_config=None,
            aux_render_modes=('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
            name='VerboseGuidedA3C',
            **kwargs
    ):
        super(VerboseGuidedAAC, self).__init__(
            name=name,
            runner_config={
                    'class_ref': VerboseSynchroRunner,
                    'kwargs': {
                        'aux_render_modes': aux_render_modes,
                    }
                },
            aux_render_modes=aux_render_modes,
            **kwargs
        )

