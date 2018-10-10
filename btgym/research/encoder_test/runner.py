import numpy as np

from btgym.algorithms.runner.synchro import BaseSynchroRunner
from btgym.algorithms.math_utils import softmax


class RegressionRunner(BaseSynchroRunner):
    """
    Runner with additional regression functionality.
    """

    def __init__(self, **kwargs):

        super(RegressionRunner, self).__init__(
            _implemented_aux_render_modes=(
                'regression',
                'regression_targets',
                'action_prob',
                'value_fn',
                'lstm_1_h',
                'lstm_2_h'
            ),
            **kwargs
        )

    def get_init_experience(self, policy, policy_sync_op=None, init_context=None, data_sample_config=None):
        """
        Starts new environment episode.

        Args:
            policy:                 policy to execute.
            policy_sync_op:         operation copying local behavioural policy params from global one
            init_context:           initial policy context for new episode.
            data_sample_config:     configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`

        Returns:
            incomplete initial experience of episode as dictionary (misses bootstrapped R value),
            next_state,
            next, policy RNN context
            action_reward
        """
        self.length = 0
        self.reward_sum = 0
        # Increment global and local episode counters:
        self.sess.run(self.policy.inc_episode)
        self.local_episode += 1

        # self.log.warning('get_init_exp() data_sample_config: {}'.format(data_sample_config))

        if data_sample_config is None:
            data_sample_config = policy.get_sample_config()

        # Pass sample config to environment (.get_sample_config() is actually aac framework method):
        init_state = self.env.reset(**data_sample_config)

        # Master worker always resets context at the episode beginning:
        # TODO: !
        # if not self.data_sample_config['mode']:
        init_context = None

        # self.log.warning('init_context_passed: {}'.format(init_context))
        # self.log.warning('state_metadata: {}'.format(state['metadata']))

        init_action = self.env.action_space.encode(self.env.get_initial_action())
        init_reward = np.asarray(0.0)

        # Update policy:
        if policy_sync_op is not None:
            self.sess.run(policy_sync_op)

        init_context = policy.get_initial_features(state=init_state, context=init_context)
        action, logits, value, next_context, init_regression = policy.act(
            init_state,
            init_context,
            init_action[None, ...],
            init_reward[None, ...],
        )
        next_state, reward, terminal, self.info = self.env.step(action['environment'])

        experience = {
            'position': {'episode': self.local_episode, 'step': self.length},
            'state': init_state,
            'action': action['one_hot'],
            'reward': reward,
            'value': value,
            'terminal': terminal,
            'context': init_context,
            'last_action': init_action,
            'last_reward': init_reward,
            'r': None,  # to be updated
            'info': self.info[-1],
        }
        # Execute user-defined callbacks to policy, if any:
        for key, callback in policy.callback.items():
            experience[key] = callback(**locals())

        # reset per-episode  counters and accumulators:
        self.ep_accum = {
            'logits': [logits],
            'value': [value],
            'context': [init_context],
            'regression': [init_regression],
            'regression_targets': [init_state['regression_targets']],
        }
        self.terminal_end = terminal
        #self.log.warning('init_experience_context: {}'.format(context))

        # Take a nap:
        self.sleep()

        return experience, next_state, next_context, action['encoded'], reward

    def get_experience(self, policy, state, context, action, reward, policy_sync_op=None):
        """
        Get single experience (possibly terminal).

        Returns:
            incomplete experience as dictionary (misses bootstrapped R value),
            next_state,
            next, policy RNN context
            action_reward
        """
        # Continue adding experiences to rollout:
        next_action, logits, value, next_context, regression = policy.act(
            state,
            context,
            action[None, ...],
            reward[None, ...],
        )

        self.ep_accum['logits'].append(logits)
        self.ep_accum['value'].append(value)
        self.ep_accum['context'].append(next_context)
        self.ep_accum['regression'].append(regression)
        self.ep_accum['regression_targets'].append(state['regression_targets'])

        # self.log.notice('context: {}'.format(context))
        next_state, next_reward, terminal, self.info = self.env.step(next_action['environment'])

        # Partially compose experience:
        experience = {
            'position': {'episode': self.local_episode, 'step': self.length},
            'state': state,
            'action': next_action['one_hot'],
            'reward': next_reward,
            'value': value,
            'terminal': terminal,
            'context': context,
            'last_action': action,
            'last_reward': reward,
            'r': None,
            'info': self.info[-1],
        }
        for key, callback in policy.callback.items():
            experience[key] = callback(**locals())

        # Housekeeping:
        self.length += 1

        # Take a nap:
        # self.sleep()

        return experience, next_state, next_context, next_action['encoded'], next_reward

    def get_ep_render(self, is_test=False):
        """
        Visualises episode environment and policy statistics.
        Relies on environmnet renderer class methods,
        so it is only valid when environment rendering is enabled (typically it is true for master runner).

        Returns:
            dictionary of images as rgb arrays

        """
        # Only render chief worker and test (slave) environment:
        # if self.task < 1 and (
        #     is_test or(
        #         self.local_episode % self.env_render_freq == 0 and not self.data_sample_config['mode']
        #     )
        # ):
        if self.task < 1 and self.local_episode % self.env_render_freq == 0:

            # Render environment (chief worker only):
            render_stat = {
                mode: self.env.render(mode)[None, :] for mode in self.env.render_modes
            }
            # Update renderings with aux:

            # ep_a_logits = self.ep_accum['logits']
            # ep_value = self.ep_accum['value']
            # ep_regression = self.ep_accum['regression']
            # self.log.notice('ep_logits shape: {}'.format(np.asarray(ep_a_logits).shape))
            # self.log.notice('ep_value shape: {}'.format(np.asarray(ep_value).shape))
            # self.log.notice('ep_regression shape: {}'.format(np.asarray(ep_regression).shape))

            # ep_regression_targets = self.ep_accum['regression_targets']
            # self.log.notice('ep_regression_targets shape: {}'.format(np.asarray(ep_regression_targets).shape))

            # Unpack LSTM states:
            rnn_1, rnn_2 = zip(*self.ep_accum['context'])
            rnn_1 = [state[0] for state in rnn_1]
            rnn_2 = [state[0] for state in rnn_2]
            c1, h1 = zip(*rnn_1)
            c2, h2 = zip(*rnn_2)

            r = np.exp(np.asarray(self.ep_accum['regression'])[:, 0, :])
            rt = np.asarray(self.ep_accum['regression_targets'])

            # Render everything implemented (doh!):
            implemented_aux_images = {
                'action_prob': self.env.renderer.draw_plot(
                    # data=softmax(np.asarray(ep_a_logits)[:, 0, :] - np.asarray(ep_a_logits).max()),
                    data=softmax(np.asarray(self.ep_accum['logits'])), #[:, 0, :]),
                    title='Episode actions probabilities',
                    figsize=(12, 4),
                    box_text='',
                    xlabel='Backward env. steps',
                    ylabel='R+',
                    line_labels=['Hold', 'Buy', 'Sell', 'Close']
                )[None, ...],
                'value_fn': self.env.renderer.draw_plot(
                    data=np.asarray(self.ep_accum['value']),
                    title='Episode Value function',
                    figsize=(12, 4),
                    xlabel='Backward env. steps',
                    ylabel='R',
                    line_labels=['Value']
                )[None, ...],
                # 'lstm_1_c': norm_image(np.asarray(c1).T[None, :, 0, :, None]),
                'lstm_1_h': self.norm_image(np.asarray(h1).T[None, :, 0, :, None]),
                # 'lstm_2_c': norm_image(np.asarray(c2).T[None, :, 0, :, None]),
                'lstm_2_h': self.norm_image(np.asarray(h2).T[None, :, 0, :, None]),
                'regression': self.env.renderer.draw_plot(
                    data=r,
                    title='Regression',
                    figsize=(12, 4),
                    xlabel='Backward env. steps',
                    ylabel='R',
                    line_labels=['regressed values'],
                )[None, ...],
                'regression_targets': self.env.renderer.draw_plot(
                    data=rt,
                    title='Regression targets',
                    figsize=(12, 4),
                    xlabel='Backward env. steps',
                    ylabel='R',
                    line_labels=['target values'],
                )[None, ...],
            }

            self.log.notice('Mean regression value: {:.6f}, target: {:.6f}, mse: {:.6f}'.format(r.mean(), rt.mean(), (((r-rt)**2).mean())**0.5))

            # Pick what has been set:
            aux_images = {summary: implemented_aux_images[summary] for summary in self.aux_render_modes}
            render_stat.update(aux_images)

        else:
            render_stat = None

        return render_stat