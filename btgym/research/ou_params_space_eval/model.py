import numpy as np
from collections import namedtuple

from btgym.datafeed.synthetic.utils import ou_mle_estimator
from btgym.datafeed.synthetic.stochastic import ornshtein_uhlenbeck_process_batch_fn
try:
    from pykalman import KalmanFilter

except ImportError:
    raise ImportError('Locally required package `pykalman` seems not be installed.')

PairModelObservation = namedtuple('PairModelObservation', ['p', 's', 'u', 'bias', 'std'])
PairModelState = namedtuple('PairModelState', ['p_mean', 'p_cov', 's_mean', 's_cov', 'u', 'bias', 'std'])


class PairFilteredModel:
    """
    Generative model of 2x 1d [potentially integrated] processes
    as filtered evolution over base Ornshtein-Uhlenbeck model parameters space.

    Uses PCA decomposition to separately model each component, motivating paper:
        Harris, D. (1997) "Principal components analysis of cointegrated time series," in Econometric Theory, 13
    """

    def __init__(self):
        """

        """
        self.track_p_filter = None
        self.track_s_filter = None

        # Observations in OU parameters state-space:
        self.obs_trajectory = []

        # Filtered estimates in OU parameters state-space:
        self.state_trajectory = []

        # Decomposition matrix for max. correlated (colorized) data:
        self.u_colored, _, _ = np.linalg.svd(np.ones([2, 2]))

        self.is_ready = False

    @staticmethod
    def base_model_estimator(x):
        """
        Returns MLE estimated parameters of OU process on given data
        Log-scales sigma and lambda.
        Args:
            x: data array

        Returns:
            mu, log_lambda, log_sigma estimates
        """
        _mu, _lambda, _sigma = ou_mle_estimator(x)
        log_psi = np.asarray(
            [_mu, np.log(np.clip(_lambda, 1e-10, None)), np.log(np.clip(_sigma, 1e-10, None))]
        ).T
        return log_psi

    @staticmethod
    def decompose(x, u=None):
        """
        Returns eigendecomposition of pair [X1, X2] and unitary decomposition matrix:
        X = U x [U.T x X]

        Args:
            x:  data of shape [2, num_points]
            u:  transition matrix [2,2] or None; if no matrix is provided - computes via SVD.

        Returns:
            x_ps:    data projection to eigenvectors (max./min. variance explained) of size [2, num_pints]
            u:       [2,2] decomp. matrix of orthonormal vectors
            bias:    [X1, X2] bias vector of size [2, 1]
            std:     [X1, X2] standard deviation vector of size [2, 1]

        """
        assert len(x.shape) == 2, 'Expected 2d array, got: {}'.format(x.shape)
        # Center and colorize data:
        bias = x.mean(axis=-1)[..., None]
        std = np.clip(x.std(axis=-1), 1e-8, None)[..., None]
        x_c = (x - bias) / std
        # Estimate and keep track of transition matrices:
        u_t, _, _ = np.linalg.svd(np.cov(x_c))
        if u is None:
            # If no matrix was provided - use current:
            u = u_t

        x_ps = np.matmul(u.T, x_c)

        return x_ps, u_t, bias, std

    @staticmethod
    def restore(x_ps, u, bias, std):
        """
        Restores data from decomposition.

        Args:
            x_ps:   data projection to SVD eigenvectors  of size [2, num_pints]
            u:      [2,2] decomposition matrix
            bias:   original data bias vector of size [2]
            std:     [X1, X2] standard deviation vector of size [2, 1]

        Returns:
            x:  data of shape [2, num_points]
        """
        # Decolorize and decenter:
        return np.matmul(u, x_ps) * std + bias

    def ready(self):
        assert self.is_ready, 'Model is not initialized. Hint: forgot to call model.reset()?'

    def reset_em(self, init_trajectory, step_size):
        """
        Estimate model initial parameters and resets filtered trajectory.

        Args:
            init_trajectory:    2d trajectory of process [X1, X2] values
                                initial filter parameters are estimated over of size [2, len]
            step_size:          uint, embedding shift-size from X[t] to X[t+1]
        """
        self.obs_trajectory = []
        self.state_trajectory = []

        assert init_trajectory.shape[-1] > 2 * step_size, 'Initial trajectory should be longer than 2x step-size'

        # Shifted trajectories:
        x0 = init_trajectory[:, :-step_size]
        x1 = init_trajectory[:, step_size:]

        # Decompose to [Price, Spread] and get tr. operators:
        xp0, u0, b0, std0 = self.decompose(x0)
        xp1, u1, b1, std1 = self.decompose(x1, u0)

        # Get Psi_price, Psi_spread, shaped [2x3] each:
        log_psi0 = self.base_model_estimator(xp0)
        log_psi1 = self.base_model_estimator(xp1)

        # Use two independent filters to track P and S streams:
        self.track_p_filter = KalmanFilter(
            initial_state_mean=log_psi0[0, :].tolist(),
            transition_covariance=1 * np.eye(3),
            n_dim_obs=3
        )

        self.track_s_filter = KalmanFilter(
            initial_state_mean=log_psi0[1, :].tolist(),
            transition_covariance=1 * np.eye(3),
            n_dim_obs=3
        )

        # Use two observations to estimate initial parameters:
        self.track_p_filter.em(np.asarray([log_psi0[0, :], log_psi1[0, :]]))
        self.track_s_filter.em(np.asarray([log_psi0[1, :], log_psi1[1, :]]))

        # Initial observation:
        self.obs_trajectory.append(
            PairModelObservation(log_psi1[0, :], log_psi1[1, :], u1, b1, std1)
        )

        # Compose initial hidden state:
        self.state_trajectory.append(
            PairModelState(log_psi1[0, :], np.zeros([3, 3]), log_psi1[1, :], np.zeros([3, 3]), u1, b1, std1)
        )
        self.u_colored = u0
        self.is_ready = True

    def reset(self, init_trajectory, obs_covariance_factor):
        """
        Estimate model initial parameters and resets filtered trajectory.

        Args:
            init_trajectory:        2d trajectory of process [X1, X2] values
                                    initial filter parameters are estimated over of size [2, len]
            obs_covariance_factor:  ufloat, Kf identity observation matrix multiplier
        """
        self.obs_trajectory = []
        self.state_trajectory = []

        assert init_trajectory.shape[-1] > 1, 'Initial trajectory should be longer than 1'

        # Decompose to [Price, Spread] and get decomp. matrix:
        xp0, u0, b0, std0 = self.decompose(init_trajectory)

        # Get Psi_price, Psi_spread, shaped [2x3] each:
        log_psi0 = self.base_model_estimator(xp0)

        # Use two independent filters to track P and S streams:
        self.track_p_filter = KalmanFilter(
            initial_state_mean=log_psi0[0, :].tolist(),
            #transition_covariance=obs_covariance_factor * np.eye(3),
            observation_covariance=obs_covariance_factor * np.eye(3),
            n_dim_obs=3
        )

        self.track_s_filter = KalmanFilter(
            initial_state_mean=log_psi0[1, :].tolist(),
            #transition_covariance=obs_covariance_factor * np.eye(3),
            observation_covariance=obs_covariance_factor * np.eye(3),
            n_dim_obs=3
        )


        # Initial observation:
        self.obs_trajectory.append(
            PairModelObservation(log_psi0[0, :], log_psi0[1, :], u0, b0, std0)
        )

        # Compose initial hidden state:
        self.state_trajectory.append(
            PairModelState(
                log_psi0[0, :], np.zeros([3, 3]), log_psi0[1, :], np.zeros([3, 3]), u0, b0, std0
            )
        )
        self.u_colored = u0
        self.is_ready = True

    def update(self, pair_trajectory, state_prev=None, u=None):
        """
        Estimates Z[t] given a trajectories of two base 1d process and previous hidden state Z[t-1]
        Args:
            pair_trajectory:    trajectory of n past base process observations X = (x[t-n], x[t-n+1], ..., x[t]),
                                where x[i] = (p1[i], p2[i]), shaped [2, n]
            state_prev:         instance of 'PairModelState' - previous state of the model, Z[t-1]
            u:                  [2,2] decomposition matrix of orthonormal vectors, typically is self.u_init, or None

        Returns:
            instance of 'PairModelState', Z[t]
        """
        self.ready()

        # Use last available Z if no state is given:
        if state_prev is None:
            z = self.state_trajectory[-1]

        else:
            assert isinstance(state_prev, PairModelState), \
                'Expected model state as instance of `PairModelState`, got: {}'.format(type(state_prev))
            z = state_prev

        # Decompose down to `base line`, `spread line` and rotating operator:
        x_decomp, u, b, std = self.decompose(pair_trajectory, u=u)

        # Estimate hidden state observation [Psi_p[t] | Psi_s[t]], shaped [2, 3]:
        log_psi = self.base_model_estimator(x_decomp)

        # Estimate Z[t] given Z[t-1] and Psi[t]:
        z_mean_p, z_cov_p = self.track_p_filter.filter_update(
            filtered_state_mean=z.p_mean,
            filtered_state_covariance=z.p_cov,
            observation=log_psi[0,:],
        )

        z_mean_s, z_cov_s = self.track_s_filter.filter_update(
            filtered_state_mean=z.s_mean,
            filtered_state_covariance=z.s_cov,
            observation=log_psi[1, :],
        )

        self.obs_trajectory.append(
            PairModelObservation(log_psi[0, :], log_psi[1, :], u, b, std)
        )
        self.state_trajectory.append(
            PairModelState(z_mean_p, z_cov_p, z_mean_s, z_cov_s, u, b, std)
        )

    @staticmethod
    def sample_state_fn(batch_size, state, mu_as_mean=True):
        """
        Generates batch of realisations of model hidden state Z.
        Static method, can be used as stand-along function.

        Args:
            batch_size:     number of sample to draw
            state:          instance of hidden state Z
            mu_as_mean:    bool, if True - return mu as value of Z_mean_mu, else return sample N(Z_mean_mu, Z_cov)

        Returns:
            Psi values as tuple: (mu[batch_size, 2], lambda[batch_size, 2], sigma[batch_size, 2])
        """
        z = state
        # Sample Psi given Z:
        log_psi_p = np.random.multivariate_normal(mean=z.p_mean, cov=z.p_cov, size=batch_size)
        log_psi_s = np.random.multivariate_normal(mean=z.s_mean, cov=z.s_cov, size=batch_size)

        log_psi = np.stack([log_psi_p, log_psi_s], axis=1)

        # Fighting the caveat: sampled mu values can cause excessive jumps; alternative is to use Z mean:
        if mu_as_mean:
            mu_mean = np.stack([z.p_mean, z.s_mean], axis=0)
            psi_mu = np.ones([batch_size, 2]) * mu_mean[:, 0]

        else:
            psi_mu = log_psi[..., 0]

        psi_lambda = np.exp(log_psi[..., 1])
        psi_sigma = np.exp(log_psi[..., 2])

        return psi_mu, psi_lambda, psi_sigma

    # @staticmethod
    # def sample_state_fn_2(batch_size, state):
    #     """
    #     Generates batch of realisations of model hidden state Z.
    #     Static method, can be used as stand-along function.
    #
    #     Args:
    #         batch_size: number of sample to draw
    #         state:      instance of hidden state Z
    #
    #     Returns:
    #         Psi values as tuple: (mu[batch_size], lambda[batch_size], sigma[batch_size])
    #     """
    #     z = state
    #     # Sample Psi given Z:
    #     log_psi = np.random.multivariate_normal(mean=z[0], cov=z[1], size=batch_size)
    #     psi_mu = log_psi[:, 0]
    #     psi_lambda = np.exp(log_psi[:, 1])
    #     psi_sigma = np.exp(log_psi[:, 2])
    #
    #     return psi_mu, psi_lambda, psi_sigma

    def generate_state(self, batch_size, state=None, mu_as_mean=True):
        """
        Generates batch of realisations of model hidden state Z.

        Args:
            batch_size:     number of sample to draw
            state:          instance of Z or None; if no state provided - last state from filtered trajectory is used
            mu_as_mean:     bool, if True - return mu as value of Z_mean_mu, else return sample N(Z_mean_mu, Z_cov)

        Returns:
            Z samples as tuple: (mu[batch_size], lambda[batch_size], sigma[batch_size])
        """
        self.ready()
        if state is None:
            z = self.state_trajectory[-1]

        else:
            assert isinstance(state, PairModelState), \
                'Expected model state as instance of `PairModelState`, got: {}'.format(type(state))
            z = state

        return self.sample_state_fn(batch_size, z, mu_as_mean)

    @staticmethod
    def generate_trajectory_fn(batch_size, num_points, state, mu_as_mean=True, restore=True, u=None):
        """
        Generates batch of realisations of pair base 1d processes given model hidden state Z in fully sequential
        fashion i.e. state value Psi is re-sampled for every X[i].
        Static method, can be used as stand-along function.

        Args:
            batch_size:     uint, number of trajectories to generates
            num_points:     uint, trajectory length to generate
            state:          model hidden state Z as (Z_means, Z_covariances) tuple.
            mu_as_mean:     bool, if True - se mu as value of Z_mean_mu, else sample same way as lambda and sigma.
            restore:        bool, if True - restore to original timeseries X1|X2, return S|P decomposition otherwise
            u:              [2,2] decomposition matrix, typically is self.u_init, or None

        Returns:
            base data process realisations as array of shape [batch_size, 2, num_points] or
            P, S projections decomposition as array of shape [batch_size, 2, num_points]
        """
        state_sampler = PairFilteredModel.sample_state_fn

        # Get init. trajectories points:
        psi_mu, _, _ = state_sampler(batch_size, state, mu_as_mean)
        xp_trajectory = [psi_mu.flatten()]
        for i in range(num_points):
            # Sample Psi given Z:
            psi_mu, psi_lambda, psi_sigma = state_sampler(batch_size, state, mu_as_mean)
            # print('m-l-s shapes: ', psi_mu.shape, psi_lambda.shape, psi_sigma.shape)

            # Sample next x[t] point given Psi and x[t-1]:
            x_next = ornshtein_uhlenbeck_process_batch_fn(
                1,
                mu=psi_mu.flatten(),
                l=psi_lambda.flatten(),
                sigma=psi_sigma.flatten(),
                x0=xp_trajectory[-1]
            )
            xp_trajectory.append(np.squeeze(x_next))

        xp_trajectory = np.asarray(xp_trajectory).T
        xp_trajectory = xp_trajectory.reshape([batch_size, 2, -1])[..., 1:]

        if restore:
            if u is None:
                x = np.matmul(state.u, xp_trajectory) * state.std + state.bias

            else:
                x = np.matmul(u, xp_trajectory) * state.std + state.bias

        else:
            x = xp_trajectory

        return x

    @staticmethod
    def generate_trajectory_fn_2(batch_size, num_points, state):
        """
        Generates batch of realisations of base 1d process given model hidden state Z.
        State value Psi sampled only once for entire trajectory.
        Static method, can be used as stand-along function.

        Args:
            batch_size: uint, number of trajectories to generates
            num_points: uint, trajectory length to generate
            state:      model hidden state Z as (Z_means, Z_covariances) tuple.

        Returns:
            base data process realisations as 2d array of shape [batch_size, num_points]
        """
        state_sampler = PairFilteredModel.sample_state_fn

        # Sample Psi given Z:
        psi_mu, psi_lambda, psi_sigma = state_sampler(batch_size, state)

        x = ornshtein_uhlenbeck_process_batch_fn(
            num_points,
            mu=psi_mu,
            l=psi_lambda,
            sigma=psi_sigma,
            x0=psi_mu
        )

        return x

    def generate_trajectory(self, batch_size, num_points, state=None, mu_as_mean=True, restore=True, u=None):
        """
        Generates batch of realisations of pair base 1d processes given model hidden state Z.

        Args:
            batch_size:     uint, number of trajectories to generates
            num_points:     uint, trajectory length to generate
            state:          instance of Z or None; if no state provided - last state from filtered trajectory is used.
            mu_as_mean:     bool, if True - se mu as value of Z_mean_mu, else sample same way as lambda and sigma.
            restore:        bool, if True - restore to original timeseries X1|X2, return S|P decomposition otherwise
            u:              [2,2] decomposition matrix, typically is self.u_init, or None

        Returns:
            Returns:
            base data process realisations as array of shape [batch_size, 2, num_points] or
            P, S projections decomposition as array of shape [batch_size, 2, num_points]
        """
        self.ready()
        if state is None:
            state = self.state_trajectory[-1]
        return self.generate_trajectory_fn(batch_size, num_points, state, mu_as_mean, restore, u)

    def generate_trajectory_2(self, batch_size, num_points, state=None, **kwargs):
        """
        Generates batch of realisations of base 1d process given model hidden state Z.

        Args:
            batch_size: uint, number of trajectories to generates
            num_points: uint, trajectory length to generate
            state:      instance of Z or None; if no state provided - last state from filtered trajectory is used.

        Returns:
            base data process realisations as 2d array of shape [batch_size, num_points]
        """
        self.ready()
        if state is None:
            state = self.state_trajectory[-1]
        return self.generate_trajectory_fn_2(batch_size, num_points, state)
