import  numpy as np

from btgym.algorithms.utils import batch_stack, batch_gather, _show_struct
from btgym.research.gps.aac import GuidedAAC
from btgym.algorithms.runner.synchro import BaseSynchroRunner


class AACt2d(GuidedAAC):
    """
    Trajectory2Distribution:
    AAC class including methods enabling treating collected train data as empirical distribution rather than trajectory.

    Note:
        time_flat=True is a key ingredient here. See BaseAAC notes for details.
    """
    def __init__(
            self,
            runner_config=None,
            name='AAC_T2D',
            **kwargs
    ):
        try:
            if runner_config is None:
                kwargs['runner_config'] = {
                'class_ref': BaseSynchroRunner,
                'kwargs': {
                    'data_sample_config': {'mode': 0},
                    'name': 't2d_synchro',
                },
            }
            else:
                kwargs['runner_config'] = runner_config
            kwargs.update(
                {
                    'time_flat': True,
                    'name': name,
                    '_aux_render_modes': ('action_prob', 'value_fn', 'lstm_1_h', 'lstm_2_h'),
                }
            )
            super(AACt2d, self).__init__(**kwargs)
            self.on_policy_batch = None
            self.off_policy_batch = None
            self.rp_batch = None

        except:
            msg = 'AAC_T2D.__init__() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def get_episode(self, **kwargs):
        """
         Get exactly one episode trajectory as single rollout. <-- DEAD WRONG

        Args:
            **kwargs:   see env.reset() method

        Returns:

        """
        data_streams = [runner.get_episode(**kwargs) for runner in self.runners]
        return {key: [stream[key] for stream in data_streams] for key in data_streams[0].keys()}

    def get_batch(self, **kwargs):
        """
        Retrieves batch of rollouts from runners.

        Args:
            **kwargs:   see runner.get_batch() method.

        Returns:
            dictionary of batched rollouts
        """
        rollouts = []
        terminal_context = []
        for runner in self.runners:
            batch = runner.get_batch(**kwargs)
            for rollout in batch['data']:
                rollouts.append(rollout)

            for context in batch['terminal_context']:
                terminal_context.append(context)

        self.log.debug('rollouts_len: {}'.format(len(rollouts)))

        final_batch = {key: [rollout[key] for rollout in rollouts] for key in rollouts[0].keys()}
        final_batch['terminal_context'] = terminal_context

        return final_batch

    @staticmethod
    def sample_batch(batch, sample_size):
        """
        Uniformly randomly samples mini-batch from (supposedly bigger) batch.

        Args:
            batch:          nested dict of experiences
            sample_size:    mini-batch size

        Returns:
            nested dict of experiences of same structure as `batch` with number of experiences eq. to `sample_size`.
        """
        if batch is not None:
            batch_size = batch['time_steps'].shape[0]
            indices = np.random.randint(0, batch_size, size=sample_size)

            return batch_gather(batch, indices)

        else:
            return  None

    def process_batch(self, sess, data):
        """
        Processes batched rollouts.
        Makes every experience independent, enabling further shuffling or sampling.

        Args:
            sess:               tf session obj.
            data (dict):        data dictionary
            is_train (bool):    is data provided are train or test

        Returns:
            on-policy [, off-policy and rp] processed batched data.

        """
        # Process minibatch for on-policy train step:
        on_policy_batch = self._process_rollouts(data['on_policy'])

        if self.use_memory:
            # Process rollouts from replay memory:
            off_policy_batch = self._process_rollouts(data['off_policy'])

            if self.use_reward_prediction:
                # Rebalanced 50/50 sample for RP:
                rp_rollouts = data['off_policy_rp']
                rp_batch = batch_stack([rp.process_rp(self.rp_reward_threshold) for rp in rp_rollouts])

            else:
                rp_batch = None

        else:
            off_policy_batch = None
            rp_batch = None

        return on_policy_batch, off_policy_batch, rp_batch

    @staticmethod
    def _check(batch):
        """
        Debug. utility.
        """
        print('Got data_dict:')
        for key in batch.keys():
            try:
                shape = np.asarray(batch[key]).shape
            except:
                shape = '???'
            print('key: {}, shape: {}'.format(key, shape))

    def process(self, sess, **kwargs):
        """
        Usage example. Override.
        """
        try:
            # Collect train trajectories:
            train_data_batch = self.get_batch(size=30, require_terminal=True)

            self._check(train_data_batch)
            #print('train_data_batch_ep_summary: ', train_data_batch['ep_summary'])
            #print('train_data_batch_render_summary: ', train_data_batch['render_summary'])

            # Process time-flat alike (~iid) to treat as empirical data distribution over train task:
            self.on_policy_batch, self.off_policy_batch, self.rp_batch = self.process_batch(sess, train_data_batch)

            #self._check(self.on_policy_batch)

            print('on_p_batch_size: {}'.format(self.on_policy_batch['batch_size']))

            # Perform updates to
            # Sample random batch of train data from train task:
            on_policy_mini_batch = self.sample_batch(self.on_policy_batch, 17)

            #self._check(on_policy_mini_batch)

            print('on_p_mini_batch_size: {}'.format(on_policy_mini_batch['batch_size']))

            feed_dict = self._get_main_feeder(sess, on_policy_mini_batch, None, None, True)

            #self._check(feed_dict)

        except:
            msg = 'process() exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)


