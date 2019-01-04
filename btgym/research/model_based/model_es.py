import numpy as np
from scipy import stats
from collections import namedtuple

from btgym.research.model_based.utils import ou_mle_estimator
from btgym.research.model_based.stochastic import ou_process_t_driver_batch_fn

from btgym.research.model_based.rec import Zscore, ZscoreState, Covariance, CovarianceState
from btgym.research.model_based.rec import SSA,  SSAState, OUEstimator, OUEstimatorState

from btgym.research.model_based.utils import batch_covariance


TimeSeriesEstimator = namedtuple('TimeSeriesEstimator', ['ssa', 'ou', 'ou_filter'])

PairState = namedtuple('PairState', ['p', 's', 'mean', 'variance'])

# TODO: stochastic process estimator wrapper: base estimator + smoothing filter
# TODO: time-series decomposition (analyzer?) wrapper: SSA + process estimator
# TODO: pair decomposition/reconstruction wrapper
# TODO: model: decomposition/reconstruction/generating

OUProcessState = namedtuple('OUProcessState', ['observation', 'filtered', 'driver_df'])


class OUProcess:
    """
    Provides essential functionality for recursive time series modeling
    as Ornshteinh-Uhlenbeck stochastic process: parameters estimation, filtering, trajectories generation;
    """
    def __init__(self, alpha=None, filter_alpha=None):
        self.alpha = alpha
        self.filter_alpha = filter_alpha
        self.estimator = OUEstimator(alpha)
        # Just use exponential smoothing as state-space trajectory filter:
        self.filter = Covariance(3, alpha=filter_alpha)

        # Driver is Student-t:
        self.driver_estimator = stats.t
        self.driver_df = None

        self.is_ready = False

    def ready(self):
        assert self.is_ready, 'OUProcess is not initialized. Hint: forgot to call .reset()?'

    def get_state(self):
        """
        Returns model state tuple.

        Returns:
            current state as instance of OUProcessState
        """
        return OUProcessState(
            observation=self.estimator.get_state(),
            filtered=self.filter.get_state(),
            driver_df=self.driver_df,
        )

    def fit_driver(self, trajectory):
        """
        Updates Student-t driver shape parameter. Needs entire trajectory for correct estimation.
        TODO: make recursive update.

        Args:
            trajectory: full observed data of size [max_length]

        Returns:
            Estimated shape parameter.
        """
        self.ready()
        if self.driver_df is None:

            self.driver_df, _, _ = self.driver_estimator.fit(trajectory)
            if self.driver_df <= 2.0:
                self.driver_df = 1e6

        return self.driver_df

    def reset(self, init_trajectory):
        """
        Resets model parameters for process dX = -Theta *(X - Mu) + Sigma * dW
        and starts new trajectory given initial data.

        Args:
            init_trajectory:    initial 1D process observations trajectory of size [num_points]
        """
        init_observation = np.asarray(self.estimator.reset(init_trajectory))

        # 2x observation to get initial covariance matrix estimate:
        init_observation = np.stack([init_observation, init_observation], axis=-1)
        _ = self.filter.reset(init_observation)

        self.driver_df = None
        self.is_ready = True

    def update(self, trajectory, disjoint=False):
        """
        Updates model parameters estimates for process dX = -Theta *(X - Mu) + Sigma * dW
        given new observations.

        Args:
            trajectory:  1D process observations trajectory update of size [num_points]
            disjoint:    bool, indicates whether update given is continuous or disjoint w.r.t. previous update
        """
        self.ready()
        # Get new state-space observation:
        observation = self.estimator.update(trajectory, disjoint)

        # Smooth and make it random variable:
        _ = self.filter.update(np.asarray(observation)[:, None])

        self.driver_df = None

    @staticmethod
    def sample_from_filtered(filter_state, size=1):
        """
        Samples process parameters values given filtered parameters distribution.
        Static method, can be used as stand-along function.

        Args:
            filter_state:  instance of CovarianceState of dimensionality 3
            size:          int or None, number of samples to draw

        Returns:
            sampled process parameters of size [size] each, packed as OUEstimatorState tuple

        """
        assert isinstance(filter_state, CovarianceState),\
            'Expected filter_state as instance of CovarianceState, got: {}'.format(type(filter_state))

        sample = np.random.multivariate_normal(filter_state.mean, filter_state.covariance, size=size)

        return OUEstimatorState(
            mu=sample[:, 0],
            log_theta=sample[:, 1],
            log_sigma=sample[:, 2],
        )

    def sample_parameters(self, state=None, size=1):
        """
        Samples process parameters values given process state;

        Args:
            state:  instance of OUProcessState or None;
                    if no state provided - current state is used;
            size:   number of samples to draw;

        Returns:
            sampled process parameters of size [size] each, packed as OUEstimatorState tuple
        """
        if state is None:
            state = self.get_state()

        else:
            assert isinstance(state, OUProcessState),\
                'Expected state as instance of OUProcessState, got: {}'.format(type(state))

        return self.sample_from_filtered(state.filtered, size=size)

    @staticmethod
    def generate_trajectory_fn_3(batch_size, size, parameters, t_df):
        """
        Generates batch of realisations of given process parameters.
        Static method, can be used as stand-along function.

        Args:
            batch_size:     uint, number of trajectories to generates
            size:           uint, trajectory length to generate
            parameters:     instance of OUEstimatorState of size [batch_size] for each parameter
            t_df:           float > 3.0, driver shape param.

        Returns:
            process realisations as 2d array of size [batch_size, size]
        """
        assert isinstance(parameters, OUEstimatorState), \
            'Expected `parameters` as instance of OUEstimatorState, got: {}'.format(type(parameters))

        for param in parameters:
            assert param.shape[0] == batch_size,\
                'Given `parameters` length: {} and `batch_size`: {} does not match.'.format(param.shape[0], batch_size)

        if isinstance(t_df, float) or isinstance(t_df, int):
            t_df = np.tile(t_df, batch_size)

        else:
            assert t_df.shape[0] == batch_size, \
                'Given `t_df` parameters length: {} and `batch_size`: {} does not match.'.format(t_df.shape[0], batch_size)

        trajectory = ou_process_t_driver_batch_fn(
            size,
            mu=parameters.mu,
            l=np.exp(parameters.log_theta),
            sigma=np.exp(parameters.log_sigma),
            df=t_df,
            x0=parameters.mu,
        )
        # trajectory = trajectory.T.reshape([batch_size, 3, -1])  # [..., 1:]

        return trajectory.T

    def generate(self, batch_size, size, state=None, driver_df=None):
        """
        Generates batch of realisations given process state.

        Args:
            batch_size:     uint, number of trajectories to generates
            size:           uint, trajectory length to generate
            state:          instance of OUProcessState or None;
                            if no state provided - current state is used.
            driver_df:      t-student process driver degree of freedom parameter or None;
                            if no value provided - current value is used;

        Returns:

        """
        self.ready()
        parameters = self.sample_parameters(state, size=batch_size)

        if driver_df is None:
            if self.driver_df is None:
                # Make it gaussian:
                t_df = 1e6

            else:
                t_df = self.driver_df

        else:
            t_df = driver_df

        return self.generate_trajectory_fn_3(batch_size, size, parameters, t_df)


TimeSeriesModelState = namedtuple('TimeSeriesState', ['process', 'analyzer'])


class TimeSeriesModel:
    """
    Time-series modeling and decomposition wrapper class.
    Basic idea is that observed data are treated as a realisation of some underlying stochastic process.

    Model class itself consist of two functional parts (both based on empirical covariance estimation):
    - stochastic process modeling (fitting and tracking of unobserved parameters, new data generation);
    - realisation trajectory analysis and decomposition;

    """

    def __init__(self, max_length, analyzer_window, analyzer_grouping=None, alpha=None, filter_alpha=None):
        """

        Args:
            max_length:         uint, maximum trajectory length to keep;
            analyzer_window:    uint, SSA embedding window;
            analyzer_grouping:  SSA decomposition triples grouping,
                                iterable of pairs convertible to python slices, i.e.:
                                grouping=[[0,1], [1,2], [2, None]];
            alpha:              float in [0, 1], SSA and process estimator decaying factor;
            filter_alpha:       float in [0, 1], process smoothing decaying factor;
        """
        self.process = OUProcess(alpha=alpha, filter_alpha=filter_alpha)
        self.analyzer = SSA(window=analyzer_window, max_length=max_length, grouping=analyzer_grouping, alpha=alpha)

    def get_state(self):
        return TimeSeriesModelState(
            process=self.process.get_state(),
            analyzer=self.analyzer.get_state(),
        )

    def ready(self):
        assert self.process.is_ready and self.analyzer.is_ready,\
            'TimeSeriesModel is not initialized. Hint: forgot to call .reset()?'

    def reset(self, *args, **kwargs):
        self.process.reset(*args, **kwargs)
        _ = self.analyzer.reset(*args, **kwargs)

    def update(self, *args, **kwargs):
        _ = self.analyzer.update(*args, **kwargs)
        self.process.update(*args, **kwargs)

    def get_decomposition(self, *args, **kwargs):
        return self.analyzer.transform(*args, **kwargs)

    def get_trajectory(self, *args, **kwargs):
        return self.analyzer.get_trajectory(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.process.generate(*args, **kwargs)


class DummyAnalyzer:
    """
    When data analysis is not required.
    Just keep that data.
    """
    def __init__(self):
        pass

    def get_trajectory(self):
        """Get stored trajectory"""
        pass
#
#
# class PairESModel:
#     """
#     New & improved:
#     t-drivers, better mixing, exponentially smoothing windows, non-isotropic covariance
#
#     Generative model for 2x 1d [potentially co-integrated] processes
#     as exponentiallt smoothed evolution over base Ornshtein-Uhlenbeck model parameters space.
#     Uses orthogonal decomposition to min./max. variance (P, S) components
#     and builds separate AR(1) model for each;
#     motivating paper:
#     Harris, D. (1997) "Principal components analysis of cointegrated time series," in Econometric Theory, 13
#     """
#     # TODO: final data non-negativity checks
#
#     # Decomposition matrix:
#     u_decomp = np.asarray([[.5, .5], [1., -1.]])
#
#     # Reconstruction (inverse u_decomp):
#     u_recon = np.asarray([[1.,  .5], [1., -.5]])
#
#     def __init__(self, ssa_depth, max_length, grouping=None, alpha=None, ou_filter_alpha=None):
#
#         """
#         Stateful model.
#         """
#         self.ssa_depth = ssa_depth
#         self.max_length = max_length
#         self.grouping = grouping
#         self.alpha = alpha
#         self.ou_filter_alpha = ou_filter_alpha
#
#         # Pair:
#         # TODO: pair wrapper + generator IS A model, uh?
#
#         # Components:
#         # TODO: make single timeseries wrapper
#         self.p = TimeSeriesEstimator(
#             ssa=SSA(self.ssa_depth, self.max_length, self.grouping, self.alpha),
#             ou=OUEstimator(self.alpha),
#             ou_filter=Covariance(3, self.ou_filter_alpha)
#         )
#
#         self.is_ready = False
#
#     @staticmethod
#     def base_process_estimator(x):
#         """
#         Returns MLE estimated parameters of OU process on given data
#         Log-scales sigma and lambda.
#         Args:
#             x: data array
#
#         Returns:
#             mu, log_lambda, log_sigma estimates
#         """
#         _mu, _theta, _sigma = ou_mle_estimator(x)
#         log_psi = np.asarray(
#             [_mu, np.log(np.clip(_theta, 1e-10, None)), np.log(np.clip(_sigma, 1e-10, None))]
#         ).T
#         return log_psi
#
#     @staticmethod
#     def decompose(x, u):
#         """
#         Returns orthogonal decomposition of pair [X1, X2] and data statistics: std, bias:
#
#         Args:
#             x:  data of shape [2, num_points]
#             u:  [2,2] decomposition matrix
#
#         Returns:
#             x_ps:    data projection to eigenvectors (max./min. variance explained) of size [2, num_pints]
#             u:       [2,2] decomp. matrix of orthonormal vectors
#             bias:    [X1, X2] bias vector of size [2, 1]
#             std:     [X1, X2] standard deviation vector of size [2, 1]
#
#         """
#         assert len(x.shape) == 2, 'Expected data as 2d array, got: {}'.format(x.shape)
#         assert len(u.shape) == 2, 'Expected U as 2x2 matrix, got: {}'.format(u.shape)
#
#         # Z-score data:
#         bias = x.mean(axis=-1)[..., None]
#         x_ub = x - bias
#         std = np.clip(x_ub.std(axis=-1), 1e-8, None)[..., None]
#
#         x_c = x_ub / std
#         x_ps = np.matmul(u, x_c)
#
#         return x_ps, bias, std
#
#     @staticmethod
#     def restore(x_ps, u, bias, std):
#         """
#         Restores data from PS decomposition.
#
#         Args:
#             x_ps:   data projections of size [2, num_pints]
#             u:      [2,2] decomposition matrix
#             bias:   original data bias vector of size [2]
#             std:     [X1, X2] standard deviation vector of size [2, 1]
#
#         Returns:
#             x:  data of shape [2, num_points]
#         """
#         # Decolorize and decenter:
#         return np.matmul(u, x_ps) * std + bias
#
#     def ready(self):
#         assert self.is_ready, 'Model is not initialized. Hint: forgot to call model.reset()?'
#
#     def reset(self, init_trajectory, obs_covariance_factor=1e-2):
#         """
#         Estimate model initial parameters and resets filtered trajectory.
#
#         Args:
#             init_trajectory:        2d trajectory of process [X1, X2] values
#                                     initial filter parameters are estimated over of size [2, len]
#             obs_covariance_factor:  ufloat, Kalman Filter identity observation matrix multiplier
#         """
#         self.obs_trajectory = []
#         self.state_trajectory = []
#
#         assert init_trajectory.shape[-1] > 1, 'Initial trajectory should be longer than 1'
#
#         # Decompose to [Price, Spread] and get decomp. matrix:
#         xp0, b0, std0 = self.decompose(init_trajectory, self.u_decomp)
#
#         # Get Psi_price, Psi_spread, shaped [2x3] each:
#         log_psi0 = self.base_process_estimator(xp0)
#
#         # Use two independent filters to track P and S streams:
#         self.track_p_filter = KalmanFilter(
#             initial_state_mean=log_psi0[0, :].tolist(),
#             #transition_covariance=obs_covariance_factor * np.eye(3),
#             observation_covariance=obs_covariance_factor * np.eye(3),
#             n_dim_obs=3
#         )
#         self.track_s_filter = KalmanFilter(
#             initial_state_mean=log_psi0[1, :].tolist(),
#             #transition_covariance=obs_covariance_factor * np.eye(3),
#             observation_covariance=obs_covariance_factor * np.eye(3),
#             n_dim_obs=3
#         )
#         # Fit SSA parameters for S and P:
#         length = xp0.shape[-1]
#         self.p_ssa_estimator = SSA(self.ssa_depth, length, self.grouping, self.alpha)
#         self.s_ssa_estimator = SSA(self.ssa_depth, length, self.grouping, self.alpha)
#
#         _, p_ssa_state = self.p_ssa_estimator.reset(xp0[0, :])
#         _, s_ssa_state = self.s_ssa_estimator.reset(xp0[1, :])
#
#         # Fit t distribution shape for  S, P values:
#         s_tdf, _, _ = stats.t.fit(xp0[1, :])
#
#         # Initial observation:
#         self.obs_trajectory.append(
#             PairESMobservation(log_psi0[0, :], log_psi0[1, :], b0, std0)
#         )
#         # Compose initial hidden state:
#         self.state_trajectory.append(
#             PairESMstate(
#                 p_mean=log_psi0[0, :],
#                 p_cov=np.zeros([3, 3]),
#                 s_mean=log_psi0[1, :],
#                 s_cov=np.zeros([3, 3]),
#                 bias=b0,
#                 std=std0,
#                 p_ssa_state=p_ssa_state,
#                 s_ssa_state=s_ssa_state,
#                 p_tdf=1e6,
#                 s_tdf=s_tdf,
#             )
#         )
#         self.is_ready = True
#
#     def update(self, pair_trajectory, state_prev=None):
#         """
#         Update model given trajectories:
#         Estimates Z[t] given a trajectories of two base 1d process and previous hidden state Z[t-1]
#         Args:
#             pair_trajectory:    trajectory of n past base process observations X = (x[t-n], x[t-n+1], ..., x[t]),
#                                 where x[i] = (p1[i], p2[i]), shaped [2, n]
#             state_prev:         instance of 'PairModelState' - previous state of the model, Z[t-1]
#
#         Returns:
#             instance of 'PairModelState', Z[t]
#         """
#         self.ready()
#
#         # Use last available Z if no state is given:
#         if state_prev is None:
#             z = self.state_trajectory[-1]
#
#         else:
#             assert isinstance(state_prev, PairESMstate), \
#                 'Expected model state as instance of `PairModelState`, got: {}'.format(type(state_prev))
#             z = state_prev
#
#         # Decompose down to `base line`, `spread line`:
#         x_decomp, b, std = self.decompose(pair_trajectory, u=self.u_decomp)
#
#         # Estimate hidden state observation [Psi_p[t] | Psi_s[t]], shaped [2, 3]:
#         log_psi = self.base_process_estimator(x_decomp)
#
#         # Estimate Z[t] given Z[t-1] and Psi[t]:
#         z_mean_p, z_cov_p = self.track_p_filter.filter_update(
#             filtered_state_mean=z.p_mean,
#             filtered_state_covariance=z.p_cov,
#             observation=log_psi[0,:],
#         )
#         z_mean_s, z_cov_s = self.track_s_filter.filter_update(
#             filtered_state_mean=z.s_mean,
#             filtered_state_covariance=z.s_cov,
#             observation=log_psi[1, :],
#         )
#         # SSA parameters for S and P; as we've got trajectories - forced to fit from scratch:
#         length = x_decomp.shape[-1]
#         self.p_ssa_estimator = SSA(self.ssa_depth, length, self.grouping, self.alpha)
#         self.s_ssa_estimator = SSA(self.ssa_depth, length, self.grouping, self.alpha)
#
#         _, p_ssa_state = self.p_ssa_estimator.reset(x_decomp[0, :])
#         _, s_ssa_state = self.s_ssa_estimator.reset(x_decomp[1, :])
#
#         s_tdf, _, _ = stats.t.fit(x_decomp[1, :])
#
#         self.obs_trajectory.append(
#             PairESMobservation(log_psi[0, :], log_psi[1, :], b, std)
#         )
#         self.state_trajectory.append(
#             PairESMstate(
#                 p_mean=z_mean_p,
#                 p_cov=z_cov_p,
#                 s_mean=z_mean_s,
#                 s_cov=z_cov_s,
#                 bias=b,
#                 std=std,
#                 p_ssa_state=p_ssa_state,
#                 s_ssa_state=s_ssa_state,
#                 p_tdf=1e6,
#                 s_tdf=s_tdf,
#             )
#         )
#
#     @staticmethod
#     def sample_state_fn(batch_size, state, mu_as_mean=True):
#         """
#         Generates batch of realisations given model hidden state Z.
#         Static method, can be used as stand-along function.
#
#         Args:
#             batch_size:     number of sample to draw
#             state:          instance of hidden state Z
#             mu_as_mean:    bool, if True - return mu as value of Z_mean_mu, else return sample N(Z_mean_mu, Z_cov)
#
#         Returns:
#             Psi values as tuple: (mu[batch_size, 2], lambda[batch_size, 2], sigma[batch_size, 2])
#         """
#         z = state
#         # Sample Psi given Z:
#         log_psi_p = np.random.multivariate_normal(mean=z.p_mean, cov=z.p_cov, size=batch_size)
#         log_psi_s = np.random.multivariate_normal(mean=z.s_mean, cov=z.s_cov, size=batch_size)
#
#         log_psi = np.stack([log_psi_p, log_psi_s], axis=1)
#
#         # Fighting the caveat: sampled mu values can cause excessive jumps; alternative is to use Z mean:
#         if mu_as_mean:
#             mu_mean = np.stack([z.p_mean, z.s_mean], axis=0)
#             psi_mu = np.ones([batch_size, 2]) * mu_mean[:, 0]
#
#         else:
#             psi_mu = log_psi[..., 0]
#
#         psi_lambda = np.exp(log_psi[..., 1])
#         psi_sigma = np.exp(log_psi[..., 2])
#
#         return psi_mu, psi_lambda, psi_sigma
#
#     @staticmethod
#     def sample_state_fn_3(batch_size, state, mu_as_mean=True):
#         """
#         Generates batch of realisations given model hidden state Z.
#         Get triplet: one sample for P and two for S.
#         Static method, can be used as stand-along function.
#
#         Args:
#             batch_size:     number of sample to draw
#             state:          instance of hidden state Z
#             mu_as_mean:    bool, if True - return mu as value of Z_mean_mu, else return sample N(Z_mean_mu, Z_cov)
#
#         Returns:
#             Psi values as tuple: (mu[batch_size, 3], lambda[batch_size, 3], sigma[batch_size, 3])
#         """
#         z = state
#         # Sample Psi given Z:
#         log_psi_p = np.random.multivariate_normal(mean=z.p_mean, cov=z.p_cov, size=batch_size)
#         log_psi_s1 = np.random.multivariate_normal(mean=z.s_mean, cov=z.s_cov, size=batch_size)
#         log_psi_s2 = np.random.multivariate_normal(mean=z.s_mean, cov=z.s_cov, size=batch_size)
#
#         log_psi = np.stack([log_psi_p, log_psi_s1, log_psi_s2], axis=1)
#
#         # Fighting the caveat: sampled mu values can cause excessive jumps; alternative is to use Z mean:
#         if mu_as_mean:
#             mu_mean = np.stack([z.p_mean, z.s_mean, z.s_mean], axis=0)
#             psi_mu = np.ones([batch_size, 3]) * mu_mean[:, 0]
#
#         else:
#             psi_mu = log_psi[..., 0]
#
#         psi_lambda = np.exp(log_psi[..., 1])
#         psi_sigma = np.exp(log_psi[..., 2])
#
#         return psi_mu, psi_lambda, psi_sigma
#
#     def generate_state(self, batch_size, state=None, mu_as_mean=True):
#         """
#         Generates batch of realisations given model hidden state Z.
#
#         Args:
#             batch_size:     number of sample to draw
#             state:          instance of Z or None; if no state provided - last state from filtered trajectory is used
#             mu_as_mean:     bool, if True - return mu as value of Z_mean_mu, else return sample N(Z_mean_mu, Z_cov)
#
#         Returns:
#             Z samples as tuple: (mu[batch_size], lambda[batch_size], sigma[batch_size])
#         """
#         self.ready()
#         if state is None:
#             z = self.state_trajectory[-1]
#
#         else:
#             assert isinstance(state, PairESMstate), \
#                 'Expected model state as instance of `PairModelState`, got: {}'.format(type(state))
#             z = state
#
#         return self.sample_state_fn(batch_size, z, mu_as_mean)
#
#     @staticmethod
#     def generate_trajectory_fn(batch_size, num_points, state, u, mu_as_mean=True, restore=True):
#         """
#         Generates batch of realisations of pair base 1d processes given model hidden state Z in fully sequential
#         fashion i.e. state value Psi is re-sampled for every X[i].
#         Static method, can be used as stand-along function.
#
#         Args:
#             batch_size:     uint, number of trajectories to generates
#             num_points:     uint, trajectory length to generate
#             state:          model hidden state Z as (Z_means, Z_covariances) tuple.
#             u:              [2,2] reconstruction matrix, typically is: PairFilteredModel.u_recon
#             mu_as_mean:     bool, if True - set mu as value of decomp.Z_mean_mu, else sample from distribution.
#             restore:        bool, if True - restore to original timeseries X1|X2, return None otherwise
#
#         Returns:
#             P, S projections decomposition as array of shape [batch_size, 2, num_points]
#             base data process realisations as array of shape [batch_size, 2, num_points] or None
#
#         """
#         state_sampler = PairFilteredModelT.sample_state_fn
#
#         # Get init. trajectories points:
#         psi_mu, _, _ = state_sampler(batch_size, state, mu_as_mean)
#         xp_trajectory = [psi_mu.flatten()]
#         for i in range(num_points):
#             # Sample Psi given Z:
#             psi_mu, psi_lambda, psi_sigma = state_sampler(batch_size, state, mu_as_mean)
#             # print('m-l-s-std shapes: ', psi_mu.shape, psi_lambda.shape, psi_sigma.shape, psi_std.shape)
#
#             # Sample next x[t] point given Psi and x[t-1]:
#             x_next = ou_process_t_driver_batch_fn(
#                 1,
#                 mu=psi_mu.flatten(),
#                 l=psi_lambda.flatten(),
#                 sigma=psi_sigma.flatten(),
#                 df=np.tile([state.p_tdf, state.s_tdf], batch_size),
#                 x0=xp_trajectory[-1]
#             )
#             xp_trajectory.append(np.squeeze(x_next))
#
#         xp_trajectory = np.asarray(xp_trajectory).T
#         xp_trajectory = xp_trajectory.reshape([batch_size, 2, -1])[..., 1:]
#
#         # Todo: 1. check tdf's; 2. put it through original SSA decomp; 3. sample three processes than mix
#         if restore:
#             x = np.matmul(u, xp_trajectory) * state.std + state.bias
#             # x = np.matmul(u, xp_trajectory) * state.std[..., None] + state.bias[..., None]
#
#         else:
#             x = None
#
#         return xp_trajectory, x
#
#     @staticmethod
#     def generate_trajectory_fn_3(batch_size, num_points, state, restore=True):
#         """
#         Generates batch of realisations of base 1d process given model hidden state Z.
#         State value Psi sampled only once for entire trajectory.
#         S compoment is mixture of two realisations.
#         Static method, can be used as stand-along function.
#
#         Args:
#             batch_size: uint, number of trajectories to generates
#             num_points: uint, trajectory length to generate
#             state:      model hidden state Z as (Z_means, Z_covariances) tuple.
#             restore:    bool, if True - restore to original timeseries X1|X2, return None otherwise
#
#         Returns:
#             base data process realisations as 2d array of shape [batch_size, num_points]
#         """
#         state_sampler = PairFilteredModelT.sample_state_fn_3
#
#         # Sample Psi given Z:
#         psi_mu, psi_lambda, psi_sigma = state_sampler(batch_size, state)
#
#         xp_trajectory = ou_process_t_driver_batch_fn(
#             num_points,
#             mu=psi_mu.flatten(),
#             l=psi_lambda.flatten(),
#             sigma=psi_sigma.flatten(),
#             df=np.tile([state.p_tdf, state.s_tdf, state.s_tdf], batch_size),
#             x0=psi_mu.flatten(),
#             )
#         xp_trajectory = xp_trajectory.T.reshape([batch_size, 3, -1])#[..., 1:]
#
#         if restore:
#             p = xp_trajectory[:, 0, :]
#
#             s1 = xp_trajectory[:, 1, :]
#             s2 = xp_trajectory[:, 2, :]
#
#             # No batched SSA. oblom. :(
#             # s1 = SSA._transform(SSA._delay_embed(s1, state.s_ssa_state.window), state.s_ssa_state).sum(axis=1)
#             # s2 = SSA._transform(SSA._delay_embed(s2, state.s_ssa_state.window), state.s_ssa_state).sum(axis=1)
#
#             # Mixing two spread realisations, scaling batch-wise to match variance:
#             mean_cov = batch_covariance(xp_trajectory[:, 1:, :]).sum(axis=(1, 2))[None, :]
#             rho = ((s1 - s2).var(axis=-1) / mean_cov * (2 ** .5)).T
#
#             x = np.stack([p + .5 * rho * s1, p - .5 * rho * s2], axis=1)
#
#             x = x * state.std + state.bias
#
#         else:
#             x = None
#
#         return xp_trajectory, x
#
#     def generate_trajectory(self, batch_size, num_points, state=None, mu_as_mean=True, restore=True):
#         """
#         Generates batch of realisations of pair base 1d processes given model hidden state Z.
#
#         Args:
#             batch_size:     uint, number of trajectories to generates
#             num_points:     uint, trajectory length to generate
#             state:          instance of Z or None; if no state provided - last state from filtered trajectory is used.
#             mu_as_mean:     bool, if True - set mu as value of decomp.Z_mean_mu, else sample from distribution.
#             restore:        bool, if True - restore to original timeseries X1|X2, return S|P decomposition otherwise
#
#         Returns:
#             P, S realisations decomposition as array of shape [batch_size, 2, num_points]
#             base data process realisations as array of shape [batch_size, 2, num_points] or None
#         """
#         self.ready()
#         if state is None:
#             state = self.state_trajectory[-1]
#         return self.generate_trajectory_fn(batch_size, num_points, state, self.u_recon, mu_as_mean, restore)
#
#     def generate_trajectory_3(self, batch_size, num_points, state=None, restore=True, **kwargs):
#         """
#         Generates batch of realisations of base 1d process given model hidden state Z.
#
#         Args:
#             batch_size: uint, number of trajectories to generates
#             num_points: uint, trajectory length to generate
#             state:      instance of Z or None; if no state provided - last state from filtered trajectory is used.
#             restore:    bool, if True - restore to original timeseries X1|X2, return S|P decomposition otherwise
#
#         Returns:
#             P, S1, S2 realisations decomposition as array of shape [batch_size, 3, num_points]
#             base data process realisations as array of shape [batch_size, 2, num_points] or None
#         """
#         self.ready()
#         if state is None:
#             state = self.state_trajectory[-1]
#         return self.generate_trajectory_fn_3(batch_size, num_points, state, restore)
#
