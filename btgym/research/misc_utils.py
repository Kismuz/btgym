
class EnvRunner:
    """
    Handy data provider. Runs specified environments, gets per episode data and statistic.
    """
    def __init__(self, env):
        """

        Args:
            env: btgym environment instance
        """
        self.env = env
        self.done = True
        self.sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=0,
                b_alpha=1.0,
                b_beta=1.0
            ),
            trial_config=dict(
                get_new=True,
                sample_type=0,
                b_alpha=1.0,
                b_beta=1.0
            )
        )

    def get_batch(self, batch_size):
        obs_list = []
        # batch_int = []
        batch_r = []
        batch_i = []
        while len(batch_r) < batch_size:
            if not self.done:
                o, r, self.done, i = self.env.step(self.env.get_initial_action())
            else:
                o = self.env.reset()
                r = 0
                i = None
                self.done = False
            # obs_list.append(o['raw_state'])
            # obs_list.append(o['external'])
            # batch_int.append(o['internal'])
            # batch_r.append(r)
            # batch_i.append(i)
        return obs_list

    def get_episode(self, sample_type=0):
        self.sample_config['episode_config']['sample_type'] = sample_type
        self.sample_config['trial_config']['sample_type'] = sample_type
        self.done = False
        _ = self.env.reset(**self.sample_config)
        obs_list = []
        batch_r = []
        batch_i = []
        while not self.done:
            o, r, self.done, i = self.env.step(self.env.action_space.sample())
            obs_list.append(o)
            batch_r.append(r)

        batch_obs = {key: [] for key in o.keys()}

        for obs in obs_list:
            for k, v in obs.items():
                batch_obs[k].append(v)

        batch_obs['reward'] = batch_r

        print('env_stat:\n{}'.format(self.env.get_stat()))

        return batch_obs

    def close(self):
        self.env.close()
        self.done = True