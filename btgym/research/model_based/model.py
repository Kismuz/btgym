import numpy as np
from scipy import stats
from collections import namedtuple

from btgym.research.model_based.utils import ou_mle_estimator
from btgym.research.model_based.stochastic import ornshtein_uhlenbeck_process_batch_fn
try:
    from pykalman import KalmanFilter

except ImportError:
    raise ImportError('Locally required package `pykalman` seems not be installed.')

PairModelObservation = namedtuple(
    'PairModelObservation',
    ['p', 's', 'bias', 'std']
)
PairModelState = namedtuple(
    'PairModelState',
    ['p_mean', 'p_cov', 's_mean', 's_cov', 'bias', 'std_mean', 'std_cov']
)
SpreadModelState = namedtuple(
    'SpreadModelState',
    ['spread_norm', 'd_mid_norm', 'spread_mean', 'spread_covariance', 'mid_t_params']
)
ModelState = namedtuple('ModelState', ['spread1', 'spread2', 'decomposition'])


class PairFilteredModel:
    """
    Generative model for 2x 1d [potentially co-integrated] processes
    as filtered evolution over base Ornshtein-Uhlenbeck model parameters space.
    Uses PCA-like decomposition to min./max. variance (P, S) components
    and builds separate AR(1) model for each;
    motivating paper:
    Harris, D. (1997) "Principal components analysis of cointegrated time series," in Econometric Theory, 13
    """
    # TODO: Z_mu <- narrower distr. <+ final data non-negativity checks

    # Zero degree rotation matrix:
    u_colored = np.asarray([[1.0, 1.0], [1.0, -1.0]])

    def __init__(self):
        """
        Stateful model.
        """
        self.track_p_filter = None
        self.track_s_filter = None
        self.track_std_filter = None

        # Observations in OU parameters state-space:
        self.obs_trajectory = []

        # Filtered estimates in OU parameters state-space:
        self.state_trajectory = []

        # Decomposition matrix for max. correlated (colorized) data:
        # self.u_colored, _, _ = np.linalg.svd(np.ones([2, 2]))

        # Zero degree rotation:
        # self.u_colored = np.asarray([[1.0, 1.0], [1.0, -1.0]])

        self.svd_trajectory = None

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
    def decompose(x, u):
        """
        Returns eigendecomposition of pair [X1, X2] and data statistics: std, bias:
        X = U x [U.T x X]

        Args:
            x:  data of shape [2, num_points]
            u:  [2,2] decomp. matrix of orthonormal vectors

        Returns:
            x_ps:    data projection to eigenvectors (max./min. variance explained) of size [2, num_pints]
            u:       [2,2] decomp. matrix of orthonormal vectors
            bias:    [X1, X2] bias vector of size [2, 1]
            std:     [X1, X2] standard deviation vector of size [2, 1]

        """
        assert len(x.shape) == 2, 'Expected data as 2d array, got: {}'.format(x.shape)
        assert len(u.shape) == 2, 'Expected U as 2x2 matrix, got: {}'.format(u.shape)

        # Center and colorize data:
        bias = x.mean(axis=-1)[..., None]
        x_ub = x - bias
        std = np.clip(x_ub.std(axis=-1), 1e-8, None)[..., None]
        x_c = x_ub / std

        u_, s_, v_ = np.linalg.svd(np.cov(x_c))

        x_ps = np.matmul(u.T, x_c)

        return x_ps, bias, std, (u_, s_, v_)

    @staticmethod
    def restore(x_ps, u, bias, std):
        """
        Restores data from PS decomposition.

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

    def reset(self, init_trajectory, obs_covariance_factor=1e-2):
        """
        Estimate model initial parameters and resets filtered trajectory.

        Args:
            init_trajectory:        2d trajectory of process [X1, X2] values
                                    initial filter parameters are estimated over of size [2, len]
            obs_covariance_factor:  ufloat, Kalman Filter identity observation matrix multiplier
        """
        self.obs_trajectory = []
        self.state_trajectory = []
        self.svd_trajectory = []

        assert init_trajectory.shape[-1] > 1, 'Initial trajectory should be longer than 1'

        # Decompose to [Price, Spread] and get decomp. matrix:
        xp0, b0, std0, svd0 = self.decompose(init_trajectory, self.u_colored)

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
        # TODO:
        self.track_std_filter = KalmanFilter(
            initial_state_mean=std0[:, 0].tolist(),
            #transition_covariance=obs_covariance_factor * np.eye(2),
            observation_covariance=obs_covariance_factor * np.eye(2),
            n_dim_obs=2
        )
        # Initial observation:
        self.obs_trajectory.append(
            PairModelObservation(log_psi0[0, :], log_psi0[1, :], b0, std0)
        )
        # Compose initial hidden state:
        self.state_trajectory.append(
            PairModelState(
                log_psi0[0, :], np.zeros([3, 3]), log_psi0[1, :], np.zeros([3, 3]), b0, std0[:, 0], np.zeros([2,2])
            )
        )
        self.svd_trajectory.append(svd0)
        self.is_ready = True

    def update(self, pair_trajectory, state_prev=None):
        """
        Estimates Z[t] given a trajectories of two base 1d process and previous hidden state Z[t-1]
        Args:
            pair_trajectory:    trajectory of n past base process observations X = (x[t-n], x[t-n+1], ..., x[t]),
                                where x[i] = (p1[i], p2[i]), shaped [2, n]
            state_prev:         instance of 'PairModelState' - previous state of the model, Z[t-1]

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

        # Decompose down to `base line`, `spread line`:
        x_decomp, b, std, svd = self.decompose(pair_trajectory, u=self.u_colored)

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
        # TODO: Kf -> std = {std_mean, std_cov}
        z_mean_std, z_cov_std = self.track_std_filter.filter_update(
            filtered_state_mean=z.std_mean,
            filtered_state_covariance=z.std_cov,
            observation=std[:, 0],
        )

        self.obs_trajectory.append(
            PairModelObservation(log_psi[0, :], log_psi[1, :], b, std)
        )
        self.state_trajectory.append(
            PairModelState(z_mean_p, z_cov_p, z_mean_s, z_cov_s, b, z_mean_std, z_cov_std)
        )
        self.svd_trajectory.append(svd)

    @staticmethod
    def sample_state_fn(batch_size, state, mu_as_mean=True):
        """
        Generates batch of realisations given model hidden state Z.
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
        psi_std = np.random.multivariate_normal(mean=z.std_mean, cov=z.std_cov, size=batch_size)

        log_psi = np.stack([log_psi_p, log_psi_s], axis=1)

        # Fighting the caveat: sampled mu values can cause excessive jumps; alternative is to use Z mean:
        if mu_as_mean:
            mu_mean = np.stack([z.p_mean, z.s_mean], axis=0)
            psi_mu = np.ones([batch_size, 2]) * mu_mean[:, 0]

        else:
            psi_mu = log_psi[..., 0]

        psi_lambda = np.exp(log_psi[..., 1])
        psi_sigma = np.exp(log_psi[..., 2])

        return psi_mu, psi_lambda, psi_sigma, psi_std[..., None]

    def generate_state(self, batch_size, state=None, mu_as_mean=True):
        """
        Generates batch of realisations given model hidden state Z.

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
    def generate_trajectory_fn(batch_size, num_points, state, u, mu_as_mean=True, restore=True):
        """
        Generates batch of realisations of pair base 1d processes given model hidden state Z in fully sequential
        fashion i.e. state value Psi is re-sampled for every X[i].
        Static method, can be used as stand-along function.

        Args:
            batch_size:     uint, number of trajectories to generates
            num_points:     uint, trajectory length to generate
            state:          model hidden state Z as (Z_means, Z_covariances) tuple.
            u:              [2,2] decomposition matrix, typically is: PairFilteredModel.u_colored
            mu_as_mean:     bool, if True - set mu as value of decomp.Z_mean_mu, else sample from distribution.
            restore:        bool, if True - restore to original timeseries X1|X2, return None otherwise

        Returns:
            P, S projections decomposition as array of shape [batch_size, 2, num_points]
            base data process realisations as array of shape [batch_size, 2, num_points] or None

        """
        state_sampler = PairFilteredModel.sample_state_fn

        # Get init. trajectories points:
        psi_mu, _, _, psi_std = state_sampler(batch_size, state, mu_as_mean)
        xp_trajectory = [psi_mu.flatten()]
        for i in range(num_points):
            # Sample Psi given Z:
            psi_mu, psi_lambda, psi_sigma, _ = state_sampler(batch_size, state, mu_as_mean)
            # print('m-l-s-std shapes: ', psi_mu.shape, psi_lambda.shape, psi_sigma.shape, psi_std.shape)

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
            # x = np.matmul(u, xp_trajectory) * state.std + state.bias
            x = np.matmul(u, xp_trajectory) * psi_std + state.bias

        else:
            x = None

        return xp_trajectory, x

    # @staticmethod
    # def generate_trajectory_fn_2(batch_size, num_points, state):
    #     """
    #     Generates batch of realisations of base 1d process given model hidden state Z.
    #     State value Psi sampled only once for entire trajectory.
    #     Static method, can be used as stand-along function.
    #
    #     Args:
    #         batch_size: uint, number of trajectories to generates
    #         num_points: uint, trajectory length to generate
    #         state:      model hidden state Z as (Z_means, Z_covariances) tuple.
    #
    #     Returns:
    #         base data process realisations as 2d array of shape [batch_size, num_points]
    #     """
    #     state_sampler = PairFilteredModel.sample_state_fn
    #
    #     # Sample Psi given Z:
    #     psi_mu, psi_lambda, psi_sigma = state_sampler(batch_size, state)
    #
    #     x = ornshtein_uhlenbeck_process_batch_fn(
    #         num_points,
    #         mu=psi_mu,
    #         l=psi_lambda,
    #         sigma=psi_sigma,
    #         x0=psi_mu
    #     )
    #
    #     return x

    def generate_trajectory(self, batch_size, num_points, state=None, mu_as_mean=True, restore=True):
        """
        Generates batch of realisations of pair base 1d processes given model hidden state Z.

        Args:
            batch_size:     uint, number of trajectories to generates
            num_points:     uint, trajectory length to generate
            state:          instance of Z or None; if no state provided - last state from filtered trajectory is used.
            mu_as_mean:     bool, if True - set mu as value of decomp.Z_mean_mu, else sample from distribution.
            restore:        bool, if True - restore to original timeseries X1|X2, return S|P decomposition otherwise

        Returns:
            P, S projections decomposition as array of shape [batch_size, 2, num_points]
            base data process realisations as array of shape [batch_size, 2, num_points] or None
        """
        self.ready()
        if state is None:
            state = self.state_trajectory[-1]
        return self.generate_trajectory_fn(batch_size, num_points, state, self.u_colored, mu_as_mean, restore)

    # def generate_trajectory_2(self, batch_size, num_points, state=None, **kwargs):
    #     """
    #     Generates batch of realisations of base 1d process given model hidden state Z.
    #
    #     Args:
    #         batch_size: uint, number of trajectories to generates
    #         num_points: uint, trajectory length to generate
    #         state:      instance of Z or None; if no state provided - last state from filtered trajectory is used.
    #
    #     Returns:
    #         base data process realisations as 2d array of shape [batch_size, num_points]
    #     """
    #     self.ready()
    #     if state is None:
    #         state = self.state_trajectory[-1]
    #     return self.generate_trajectory_fn_2(batch_size, num_points, state)


class HighLowSpreadModel:
    """
    Modells `High/Low` (bid/ask) spread dynamics at step `t`
    as normally distributed random variable conditioned on one-step data increments.
    Fits `Open` dynamics as t-distributed univariate random variable.
    """
    def __init__(self):
        """
        Stateless model.
        """
        pass

    @staticmethod
    def conditioned_bivariate_normal_sample(x1, mean, cov):
        """
        For a bivariate normally distributed r.v. X=(x0,x1) following N(mean, cov)
        returns values of x0 given values of x1: x0 ~ N(mean, cov|x1)

        Args:
            x1:   1d conditioning data of size [batch_size, n]
            mean: vector of means of size [2]
            cov:  covariance matrix of size [2, 2]

        Returns:
            x2 values of size [batch_size, n]
        """
        mean0_c1 = mean[0] + cov[0, 1] / cov[1, 1] * (x1 - mean[1])
        var0_c1 = cov[0, 0] - cov[0, 1] ** 2 / cov[1, 1]
        return np.random.normal(mean0_c1, var0_c1 ** .5, )  # size=x1.shape)

    @staticmethod
    def fit(data_open, data_high, data_low):
        """
        Estimates model parameters for given data.

        Args:
            data_open:      data of size [batch_dim, num_points] or [num_points] or None
            data_high:      data of size [batch_dim, num_points] or [num_points]
            data_low:       data of size [batch_dim, num_points] or [num_points]

        Returns:
            data_mid:   data vector of size [batch_dim, num_points] or [num_points]
            state:      instance of SpreadModelState, model estimated parameters
        """
        open_line = None
        assert data_low.shape == data_high.shape

        assert (data_low <= data_high).all()

        if data_open is not None:
            assert data_open.shape == data_high.shape
            assert (data_open <= data_high).all()
            assert (data_low <= data_open).all()

        if len(data_low.shape) == 1:
            low_line = data_low[None, :]
            high_line = data_high[None, :]
            if data_open is not None:
                open_line = data_open[None, :]
        else:
            low_line = data_low
            high_line = data_high
            if data_open is not None:
                open_line = data_open

        # Resulting data line of:
        data_mid = (high_line + low_line) / 2
        dmid = np.diff(data_mid, axis=-1)
        # dmid = np.concatenate([dmid[:, 0][None, :], dmid], axis=-1)
        dmid = np.concatenate([dmid[:, 0][:, None], dmid], axis=-1)

        # Model value of interest:
        d_spread = (high_line - low_line) / 2

        # Normalized data:
        spread_norm = np.clip(d_spread.max(), 1e-10, None)
        d_mid_norm = np.clip(abs(dmid).max(), 1e-10, None)

        ns = d_spread / spread_norm
        nm = abs(dmid) / d_mid_norm

        # Transformed random variable,
        # let say it loosly follows normal distribution (not actually; but
        # defined this way, resulting model is worse than reality so it is ok):
        # TODO: make it conditioned bivariate Inverse Gaussian
        vx0 = ns - nm

        # If batch is given - average across entire batch:
        spread_mean = np.asarray([vx0.mean(), nm.mean()])
        spread_covariance = np.cov(np.reshape(vx0, [-1, 1]), np.reshape(nm, [-1, 1]), rowvar=False)

        # Fit differences in `open` and `mid` to follow Student-t:
        if open_line is not None:
            df, loc, scale = stats.t.fit(np.reshape(open_line - data_mid, [-1, 1]))
            mid_t_params = dict(df=df, loc=loc, scale=scale)

        else:
            mid_t_params = None

        z = SpreadModelState(
            spread_norm=spread_norm,
            d_mid_norm=d_mid_norm,
            spread_mean=spread_mean,
            spread_covariance=spread_covariance,
            mid_t_params=mid_t_params,
        )
        # TODO: Kalman filtered z is overkill?
        return np.squeeze(data_mid), z

    @staticmethod
    def sample(data_mid, state, stochastic_open=True):
        """
        Generates Open/Hi/Low values given Mid_values conditioned on model state vector:
        High/Low  spread N-distribution parameters (mean and covariance matrix) and
        parameters of t-distributed offsets for Open values

        Args:
            data_mid:           data of size [batch_dim, n_points]
            state:                  instance of SpreadModelState holding `High/Low` model
                                parameters (mean, covariance matrix, normalisation constants) and
                                parameters of t-distributed offsets for `Open` model;
            stochastic_open:    bool, if True - use 'Open' model to generate values, return `data_mid` otherwise

        Returns:
            data of size [batch_dim, n_points, 3]
        """
        assert isinstance(state, SpreadModelState), 'Expexted `z` as instance of SpreadModelState, got: {}'.format(type(state))

        data_mid = np.squeeze(data_mid)  # paranoid: if got [n, 1] shaped data
        if len(data_mid.shape) == 1:
            data_mid = data_mid[None, :]

        dmid = np.diff(data_mid, axis=-1)
        dmid = np.concatenate([dmid[:, 0][:, None], dmid], axis=-1)
        # dmid = np.concatenate([dmid[:, 0][None, :], dmid], axis=-1)

        # Transformed conditioning x1 (scaled absolute value of t-1 `mid` difference):
        nm = abs(dmid) / state.d_mid_norm

        # Conditioned transformed sample:
        x0_cond_dmid = HighLowSpreadModel.conditioned_bivariate_normal_sample(nm, state.spread_mean, state.spread_covariance)

        # Inverse transform:
        d_spread = abs(x0_cond_dmid + nm) * state.spread_norm

        # Make `high`, `low` lines:
        high_line = data_mid + d_spread
        low_line = data_mid - d_spread

        if state.mid_t_params is not None and stochastic_open:
            # Generate t-distributed offsets from `mid` line:
            s = stats.t.rvs(size=d_spread.shape, **state.mid_t_params)

            # Force outliers to stay within high/low interval:
            trunc_idx = abs(s) > d_spread
            s[trunc_idx] = d_spread[trunc_idx] * np.sign(s[trunc_idx])

            # Compose `open` line:
            open_line = data_mid + s

            return np.stack([open_line, high_line, low_line], axis=-1)

        else:
            return np.stack([data_mid, high_line, low_line], axis=-1)


class PairDataModel:
    """
    Domain specific hard-coded probabilistic encoder/decoder.
    User-level class. Wraps `bid-ask spread` and orthogonal components decomposition models.

    Brief:
    Given a pair of matching Open/High/Low (OHL) data lines builds single generative probabilistic state-space model:

    1. Pair OHL dataset of size [2, n, 3] is transformed into single-valued 'mean' pair set of size [2, n, 1]
    and 'OHL model' parameters vectors: `High/Low` (bid/ask) spread dynamics modeled as
    normally distributed random variable conditioned on one-step data increments; `Open` dynamics modelled as
    univariate t-distributed random variable.

    2. Pair of mid-price data of size [2, n, 1] is decomposed into two orthogonal components (of size [n, 2]) by
    means of maximum and minimum explained variance of a pair (minimum and maximum stationarity components accordingly).

    3. Each component is separately modelled as Ornshtein-Uhlenbeck stochastic process giving
    single joint state-space observation.

    4. Given series of consecutive inputs of OHL data, trajectory of observations is formed and model hidden vector Z
    is obtained as Normal distributions over MLE estimated OU parameters via Kalman filtering.

    5. Given state vector Z, required amount of OHL data can be generated (decoded) by:
        - sampling from distribution over possible realisations of OU processes | Z;
        - inverse transformation from principal components to pair of 'mean' values;
        - generating OHL values by inverse applying `bid-ask spread` model.
    """
    def __init__(self, decomp_model_ref=PairFilteredModel, spread_model_ref=HighLowSpreadModel):
        """
        Stateful model.
        """
        self.spread_model = spread_model_ref()  # stateless
        self.decomp_model = decomp_model_ref()  # stateful

        self.state_trajectory = None

    @staticmethod
    def data_to_dict(data):
        """
        Casts numpy data array to dictionary of 1d arrays
        Args:
            data: np.ndarray of size [num_points, 3] or [num_points, 2] or

        Returns:
            dict. with keys {'high', 'low', ['open']} holding [num_points] sized data.
        """
        map2 = {0: 'data_high', 1: 'data_low'}
        map3 = {0: 'data_open', 1: 'data_high', 2: 'data_low'}

        assert len(data.shape) == 2

        if data.shape[-1] == 2:
            row_map = map2

        elif data.shape[-1] == 3:
            row_map = map3

        else:
            raise ValueError('Expected input array first dim. be in [2, 3], got array shaped: {}'.format(data.shape))

        data_dict = {}
        for i in range(data.shape[-1]):
            data_dict[row_map[i]] = data[:, i]

        return data_dict

    @staticmethod
    def dict_to_data(data_dict):
        """
        Casts dictionary of three arrays of size[n] to np.ndarray of size [3, n] following OHL pattern.

        Args:
            data_dict:  dict. containing keys {'high', 'low', 'open'} holding [num_points] sized data each.

        Returns:
            np.ndarray of size [3, num_points]
        """
        row_map = {0: 'data_open', 1: 'data_high', 2: 'data_low'}

        assert set(row_map.keys()) == set(data_dict.keys())

        return np.concatenate([data_dict[row_map[i]] for i in range(3)], axis=0)

    def reset(self, data1, data2, obs_covariance_factor=1e-2):
        """
        Estimate model initial parameters and resets filtered parameters trajectory.

        Args:
            data1:                  initial data 1 of size [n, 3] presenting OHL values, or [n, 2] presenting HL values
            data2:                  initial data 2, size matching data 1
            obs_covariance_factor:  ufloat, Kalman Filter identity observation matrix multiplier
        """
        self.state_trajectory = []

        data_dict1 = self.data_to_dict(data1)
        data_dict2 = self.data_to_dict(data2)

        # Infer spread model params:
        data1_mid, data1_sz = self.spread_model.fit(**data_dict1)
        data2_mid, data2_sz = self.spread_model.fit(**data_dict2)

        self.decomp_model.reset(
            init_trajectory=np.stack([data1_mid, data2_mid], axis=0),
            obs_covariance_factor=obs_covariance_factor
        )
        self.state_trajectory.append(
            ModelState(
                data1_sz,
                data2_sz,
                self.decomp_model.state_trajectory[-1]
            )
        )

    def update(self, data1, data2, state_prev=None):
        """

        Args:
            data1:          data 1 of size [n, 3] presenting OHL values, or [n, 2] presenting HL values
            data2:          initial data 2, size matching data 1
            state_prev:     `ModelState` instance, model state at t-1

        Returns:

        """
        self.decomp_model.ready()
        if state_prev is None:
            state_prev = self.state_trajectory[-1]

        else:
            assert isinstance(state_prev, ModelState), \
                'Expected model state as instance of `ModelState`, got: {}'.format(type(state_prev))

        data_dict1 = self.data_to_dict(data1)
        data_dict2 = self.data_to_dict(data2)

        # Get spread model params:
        data1_mid, data1_sz = self.spread_model.fit(**data_dict1)
        data2_mid, data2_sz = self.spread_model.fit(**data_dict2)

        # Get orthogonal decomposition:
        self.decomp_model.update(
            pair_trajectory=np.stack([data1_mid, data2_mid], axis=0),
            state_prev=state_prev.decomposition
        )
        self.state_trajectory.append(
            ModelState(
                data1_sz,
                data2_sz,
                self.decomp_model.state_trajectory[-1]
            )
        )

    def sample(self, batch_size, num_points, state=None, mu_as_mean=True, restore=True, stochastic_open=True):
        """
        Generates batch of realisations of pair base OHL processes given model hidden state.

        Args:
            batch_size:     uint, number of trajectories pairs to generate
            num_points:     uint, trajectory length to generate
            state:          instance of `ModelState` or None;
                            if no state provided - last state from filtered trajectory is used.
            mu_as_mean:     bool, if True - set mu as value of decomp.Z_mean_mu, else sample from distribution.
            restore:        bool, if True - restore to original timeseries X1|X2, return S|P decomposition otherwise

        Returns:
            P, S decomposition projections as array of shape [batch_size, 2, num_points]
            base data process OHL realisations as array of shape [batch_size, 2, num_points, 3] or None

        """
        self.decomp_model.ready()
        if state is None:
            state = self.state_trajectory[-1]

        else:
            assert isinstance(state, ModelState), \
                'Expected model state as instance of `ModelState`, got: {}'.format(type(state))

        decomposed_batch, data_mid = self.decomp_model.generate_trajectory(
            batch_size=batch_size,
            num_points=num_points,
            state=state.decomposition,
            mu_as_mean=mu_as_mean,
            restore=restore,
        )
        if data_mid is not None:
            x1_ohl = self.spread_model.sample(data_mid[:, 0, :], state=state.spread1, stochastic_open=stochastic_open)
            x2_ohl = self.spread_model.sample(data_mid[:, 1, :], state=state.spread2, stochastic_open=stochastic_open)

            if len(x1_ohl.shape) < 3:
                axis = 0

            else:
                axis = 1

            ohl_batch = np.stack([x1_ohl, x2_ohl], axis=axis)

        else:
            ohl_batch = None

        return decomposed_batch, ohl_batch






