import numpy as np
from btgym.algorithms.runner.synchro import BaseSynchroRunner


class OUpRunner(BaseSynchroRunner):
    """
    Extends `BaseSynchroRunner` class with additional summaries related to Orn-Uhl. data generating process.
    """

    def __init__(self, name='OUp_synchro', **kwargs):
        super(OUpRunner, self).__init__(name=name, **kwargs)

        # True data_generating_process params:
        self.dgp_params = {key: [] for key in self.env.observation_space.shape['metadata']['generator'].keys()}
        self.dgp_dict = {key: 0 for key in self.env.observation_space.shape['metadata']['generator'].keys()}
        # print('self.dgp_params_00: ', self.dgp_params)

        self.policy.callback['dgp_params'] = self.pull_dgp_params

    @staticmethod
    def pull_dgp_params(self, **kwargs):
        """Self is ok."""
        # print('metadata: ', kwargs['experience']['state']['metadata']['generator'])
        self.dgp_dict = kwargs['experience']['state']['metadata']['generator']

    def get_train_stat(self, is_test=False):
        """
        Updates and computes average statistics for train episodes.
        Args:
            is_test: bool, current episode type

        Returns:
            dict of stats
        """
        ep_stat = {}
        if not is_test:
            self.total_r += [self.reward_sum]
            episode_stat = self.env.get_stat()  # get episode statistic
            last_i = self.info[-1]  # pull most recent info
            self.cpu_time += [episode_stat['runtime'].total_seconds()]
            self.final_value += [last_i['broker_value']]
            self.total_steps += [episode_stat['length']]

            for key, accum in self.dgp_params.items():
                accum.append(self.dgp_dict[key])

            if self.local_episode % self.episode_summary_freq == 0:
                ep_stat = dict(
                    total_r=np.average(self.total_r),
                    cpu_time=np.average(self.cpu_time),
                    final_value=np.average(self.final_value),
                    steps=np.average(self.total_steps),
                    ou_lambda=np.average(self.dgp_params['l']),
                    ou_sigma=np.average(self.dgp_params['sigma']),
                    ou_mu=np.average(self.dgp_params['mu']),
                )
                self.total_r = []
                self.cpu_time = []
                self.final_value = []
                self.total_steps = []
                self.total_steps_atari = []
                self.dgp_params = {key: [] for key in self.env.observation_space.shape['metadata']['generator'].keys()}
                # print('ep_stat: ', ep_stat)
        return ep_stat

