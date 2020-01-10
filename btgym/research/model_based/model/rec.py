# Exponentially smoothed recursive versions
# of some useful statistics and estimators
# for time-series analysis

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
import copy
from collections import namedtuple


SSAState = namedtuple(
    'SSAState',
    ['window', 'max_length', 'grouping', 'alpha', 'covariance', 'u', 'singular_values', 'mean', 'variance']
)


class SSA:
    """
    Recursive toeplitz-style Singular Spectrum Analysis estimation of one-dimensional signal
    with arbitrary consecutive updates length.

    See:
    https://en.wikipedia.org/wiki/Singular_spectrum_analysis

    Golyandina, N. et. al., "Basic Singular Spectrum Analysis and Forecasting with R", 2012,
    https://arxiv.org/abs/1206.6910

    Golyandina, N., "Singular Spectrum Analysis for time series", 2004 [in Russian]:
    http://www.gistatgroup.com/gus/ssa_an.pdf

    Dimitrios D. Thomakos, "Optimal Linear Filtering, Smoothing and Trend Extraction
    for Processes with Unit Roots and Cointegration", 2008,
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1113331
    """

    def __init__(self, window, max_length, grouping=None, alpha=None):
        """

        Args:
            window:         uint, time embedding window
            max_length:     uint, maximum embedded signal trajectory length to keep, should be > window
            grouping:       SSA decomposition triples grouping, iterable of pairs convertible to python slices, i.e.:
                            grouping=[[0,1], [1,2], [2, None]]
            alpha:          float in [0, 1], decaying factor;

            Notes:
                alpha ~ 1 / effective_window_length;
                alpha ~ 1 - forgetting_factor,  in terms of Recursive Least Squares Filter
        """
        self.window = window
        assert max_length > window,\
            'Expected max_length_to_keep > window, got {} and {}'.format(max_length, window)
        self.max_length = max_length
        self.max_length_adj = max_length - window + 1
        self.grouping = grouping
        self.x_embedded = None
        self.cov_estimator = Covariance(window, alpha=alpha)
        self.covariance = None
        self.mean = None
        self.variance = None
        self.u = None
        self.singular_values = None
        self.v = None
        self.state = None
        self.is_ready = False

    def ready(self):
        assert self.is_ready, 'SSA is not initialized. Hint: forgot to call .reset()?'

    def get_state(self):
        """
        Convenience wrapper: pack and send everything but trajectory.

        Returns:
            instance of SSAstate: named tuple holding current estimator statistics
        """
        self.state = SSAState(
            self.window,
            self.max_length,
            self.grouping,
            self.cov_estimator.stat.alpha,
            self.covariance,
            self.u,
            self.singular_values,
            self.mean,
            self.variance,
        )
        return self.state

    def reset(self, init_trajectory):
        """
        Resets estimator state and stored trajectory.

        Args:
            init_trajectory:    initial trajectory of size [init_num_points],
                                such as: length + window > init_num_points > window

        Returns:
            embedded trajectory of size [window, init_num_points - window + 1]
        """

        assert self.max_length >= init_trajectory.shape[0] > self.window, \
            'Expected initial trajectory length be in [{}, ..., {}], got: {}'.format(
                self.window + 1, self.max_length, init_trajectory.shape[0]
            )
        init_embedding = self._update_embed(init_trajectory, disjoint=True)
        self.covariance, self.mean, self.variance = self.cov_estimator.reset(init_embedding)
        self._update_svd()
        self.is_ready = True

        return init_embedding

    def update(self, trajectory, disjoint=False):
        """
        Updates estimator state and embedded trajectory.

        Args:
            trajectory: observation trajectory of size [num_points], such as: length >= num_points > 0
            disjoint:   bool, indicates whether update given is continuous or disjoint w.r.t. previous update
                        if set to True - discards embedded trajectory already being kept.

        Returns:
            embedded update trajectory of size [window, num_points]
        """

        self.ready()
        if not disjoint:
            assert self.max_length_adj >= trajectory.shape[0] > 0,\
                'Expected continuous update length be less than: {}, got: {}'.format(self.max_length, trajectory.shape[0])
        embedded_update = self._update_embed(trajectory, disjoint=disjoint)
        self.covariance, self.mean, self.variance = self.cov_estimator.update(embedded_update)
        self._update_svd()

        return embedded_update

    def transform(self, x_embedded=None, state=None, size=None):
        """
        Return SSA signal decomposition.

        Args:
            x_embedded:      lag-embedded signal of size [window, length] or None
            state:  instance of SSAstate or None
            size:   uint or None, if given - trajectory size to transform, counting from most recent observation

        Returns:
            SSA signal decomposition of given X w.r.t. state
            if no arguments provided - returns decomposition of kept trajectory;
        """
        if x_embedded is None:
            x_embedded = self.x_embedded

        else:
            assert state is not None, 'SSAstate is expected when outer X is given, but got: None'

        if state is None:
            state = self.get_state()

        if size is not None:
            assert size > self.window - 1, 'Expected `size` no less than: {} but got: {}'.format(self.window, size)
            idx = - size + self.window - 1
            if - idx > x_embedded.shape[-1]:
                idx = None

        else:
            idx = None

        return self._transform(x_embedded[:, idx:], state)

    def _update_embed(self, x, disjoint=False):
        """
        Arbitrary length update of embedding matrix.

        Args:
            x:          observation trajectory of size [num_points], such as: length >= num_points > 0
            disjoint:   bool, indicates whether update given is continuous or disjoint w.r.t. previous update
                        if set to True - discards embedded trajectory already being kept.

        Returns:
            embedded update/home/muzikin/Repos/ml_serving

        """
        assert len(x.shape) == 1, 'Expected 1d trajectory but got input shaped: {}'.format(x.shape)
        if disjoint:
            # Been told trajectory given is NOT continuous input:
            assert self.max_length >= x.shape[0] > self.window, \
                            'Expected disjoint/initial trajectory length be in [{}, ..., {}], got: {}'.format(
                                self.window + 1, self.max_length, x.shape[0]
            )
            self.x_embedded = self._delay_embed(x, self.window)
            return self.x_embedded

        else:
            head = self.x_embedded[-1, 1 - self.window:]

            upd = np.concatenate([head, x])

            upd_embedding = self._delay_embed(upd, self.window)

            truncate_idx = np.clip(
                self.x_embedded.shape[-1] + upd_embedding.shape[-1] - self.max_length_adj,
                0,
                None
            )
            self.x_embedded = np.concatenate(
                [self.x_embedded[:, truncate_idx:], upd_embedding],
                axis=1
            )
            return upd_embedding

    def _update_svd(self):
        """
        Toeplitz variant of SSA decomposition (based on covariance matrix).
        """
        self.u, self.singular_values, self.v = np.linalg.svd(self.covariance)

    @staticmethod
    def _delay_embed(x, w):
        """
        Time-embedding with window size `w` and lag 1
        """
        g = 0
        return x[(np.arange(w) * (g + 1)) + np.arange(np.max(x.shape[0] - (w - 1) * (g + 1), 0)).reshape(-1, 1)].T

    @staticmethod
    def _henkel_diag_average(x, n, window):
        """
        Computes  diagonal averaging operator D.
        Usage: D = J.T.dot(B)*s, see:
        Dimitrios D. Thomakos, `Optimal Linear Filtering, Smoothing and Trend Extraction
        for Processes with Unit Roots and Cointegration`, 2008; pt. 2.2
        """
        J = np.ones([n - window + 1, 1])
        h = x.shape[0]

        pad = np.zeros([h, h - 1])
        pad.fill(np.nan)

        padded_x = np.r_['-1', x, pad]
        s0, s1 = padded_x.strides

        B = copy.copy(
            np.lib.stride_tricks.as_strided(padded_x, [h, n], [s0 - s1, s1], writeable=False)
        )
        B = np.ma.masked_array(B, mask=np.isnan(B))
        s = 1 / np.logical_not(B.mask).sum(axis=0)
        B[B.mask] = 0.0
        return B.data, J, s

    @staticmethod
    def _transform(x, state):
        """
        Returns SSA decomposition w.r.t. given grouping.

        Args:
            x:      embedded vector
            state:  instance of `SSAstate` holding fitted decomposition parameters
        """
        assert isinstance(state, SSAState),\
            'Expected `state` be instance of SSAstate, got {}'.format(type(state))

        n = x.shape[-1] + state.window - 1

        if state.grouping is None:
            grouping = [[i] for i in range(state.u.shape[-1])]
        else:
            grouping = state.grouping

        x_comp = []
        for group in grouping:
            d_u = state.u[:, slice(*group)]
            d_x = d_u.dot(d_u.T).dot(x)
            B, J, s = SSA._henkel_diag_average(d_x.T, n, state.window)
            x_comp.append(np.squeeze(J.T.dot(B) * s))

        return np.asarray(x_comp)

    def get_trajectory(self, size=None):
        """
        Retrieve stored fragment of original time-series data.

        Args:
            size:   uint, fragment length in [1, ..., max_length] or None

        Returns:
            1d series as [ x[-size], x[-size+1], ... x[-1] ], up to length [size]
            if no size arg is given - returns entire stored trajectory, up to length [max_length]
        """
        self.ready()
        if size is not None:
            assert 1 <= size <= self.max_length, \
                'Can only retrieve from 1 up to {} last values, but got size: {}'.format(self.max_length, size)
        else:
            size = self.max_length

        v, h = self.x_embedded.shape

        if size > v + h - 1:
            size = v + h - 1

        if size <= h:
            i_1 = - size
            i_0 = -1

        else:
            i_1 = None
            i_0 = h - size - 1

        return np.concatenate([self.x_embedded[i_0:-1, 0], self.x_embedded[-1, i_1:]], axis=-1)


ZscoreState = namedtuple('ZscoreState', ['mean', 'variance'])


class Zscore:
    """
    Recursive exponentially decayed mean and variance estimation for time-series
    with arbitrary consecutive updates length.

    Paper:
    Tony Finch: "Incremental calculation of weighted mean and variance", notes, 2009
    """

    def __init__(self, dim, alpha):
        """

        Args:
            dim:        observation dimensionality
            alpha:      float, decaying factor in [0, 1]

        """
        self.dim = dim
        if alpha is None:
            self.alpha = 1
            self.is_decayed = False
        else:
            self.alpha = alpha
            self.is_decayed = True

        self.mean = None
        self.variance = None
        self.g = None
        self.dx = None
        self.num_obs = 0

    def get_state(self):
        """
        Convenience wrapper.

        Returns:
            current state as instance of ZscoreState tuple
        """
        return ZscoreState(
            mean=self.mean,
            variance=self.variance,
        )

    def reset(self, init_x):
        """
        Resets statistics estimates.

        Args:
            init_x:  np.array of initial observations of size [dim, num_init_observations]

        Returns:
            initial dimension-wise mean and variance estimates of sizes [dim, 1], [dim, 1]

        """
        if init_x is None:
            self.mean = np.zeros(self.dim)
            self.variance = np.ones(self.dim) * 1e-8
            self.g = np.zeros(self.dim)
            self.dx = np.zeros(self.dim)
            self.num_obs = 1
            if not self.is_decayed:
                self.alpha = 1

        else:
            assert init_x.shape[0] == self.dim
            self.mean = init_x.mean(axis=-1)
            self.variance = init_x.var(axis=-1)
            self.num_obs = init_x.shape[-1]

            if not self.is_decayed:
                self.alpha = 1 / (self.num_obs - 1)

        return self.mean, self.variance

    def update(self, x):
        """
        Updates statistics estimates.

        Args:
            x: np.array, partial trajectory of shape [dim, num_updating_points]

        Returns:
            current dimension-wise mean and variance estimates of sizes [dim, 1], [dim, 1]
        """
        assert len(x.shape) == 2 and x.shape[0] == self.dim

        # Update length:
        k = x.shape[-1]

        self.num_obs += k
        if not self.is_decayed:
            self.alpha = 1 / (self.num_obs - 1)

        # Mean estimation:

        # Broadcast input to [dim, update_len, update_len]:
        xx = np.tile(x[:, None, :], [1, k, 1])

        gamma = 1 - self.alpha

        # Exp. decays as powers of (1-alpha):
        g = np.cumprod(np.repeat(gamma, k))

        # Diag. matrix of decayed coeff:
        tp = toeplitz(g / gamma, r=np.zeros(k))[::-1, ::1]

        # Backward-ordered mean updates as sums of decayed inputs:
        k_step_mean_update = np.sum(xx * tp[None, ...], axis=2)  # tp expanded for sure broadcast

        # Broadcast stored value of mean to [dim, 1] and apply decay:
        k_decayed_old_mean = (np.tile(self.mean[..., None], [1, k]) * g)

        # Get backward-recursive array of mean values from (num_obs - update_len) to (num_obs):
        means = k_decayed_old_mean + self.alpha * k_step_mean_update[:, ::-1]

        # Variance estimation:

        # Get deviations of update:
        dx = x - np.concatenate([self.mean[..., None], means[:, :-1]], axis=1)

        # Get new variance value at (num_obs) point:
        k_decayed_old_var = gamma ** k * self.variance
        k_step_var_update = np.sum(g[::-1] * dx ** 2, axis=1)

        variance = k_decayed_old_var + self.alpha * k_step_var_update

        # Update current values:
        self.mean = means[:, -1]
        self.variance = variance

        # Keep g and dx:
        self.g = g
        self.dx = dx

        return self.mean, self.variance


CovarianceState = namedtuple('CovarianceState', ['covariance', 'mean', 'variance'])


class Covariance:
    """
    Recursive exponentially decaying mean, variance and covariance matrix estimation for time-series
    with arbitrary consecutive updates length.
    """

    def __init__(self, dim, alpha=None):
        """

        Args:
            dim:        observation dimensionality
            alpha:      float, decaying factor in [0, 1]

        """
        self.stat = Zscore(dim, alpha)
        self.covariance = None
        self.mean = self.stat.mean
        self.variance = self.stat.variance

    def get_state(self):
        """
        Convenience wrapper.

        Returns:
            current state as instance of CovarianceState tuple
        """
        return CovarianceState(
            covariance=self.covariance,
            mean=self.mean,
            variance=self.variance,
        )

    def reset(self, init_x):
        """
        Resets statistics estimates.

        Args:
            init_x:   np.array of initial observations of size [dim, num_init_observations]

        Returns:
            initial covariance matrix estimate of size [dim, dim]
            initial dimension-wise means and variances of sizes [dim, 1], [dim, 1]

        """
        self.mean, self.variance = self.stat.reset(init_x)

        if init_x is None:
            self.covariance = np.eye(self.stat.dim)

        else:
            self.covariance = np.cov(init_x)

        return self.covariance, self.mean, self.variance

    def update(self, x):
        """
        Updates statistics estimates.

        Args:
            x: np.array, partial trajectory of shape [dim, num_updating_points]

        Returns:
            current covariance matrix estimate of size [dim, dim]
            current dimension-wise means and variances of sizes [dim, 1], [dim, 1]

        """
        k = x.shape[-1]
        self.mean, self.variance = self.stat.update(x)
        dx = self.stat.dx.T

        g = self.stat.g

        k_decayed_covariance = (1 - self.stat.alpha) ** k * self.covariance

        k_step_update = np.sum(g[::-1, None, None] * np.matmul(dx[:, :, None], dx[:, None, :]), axis=0)

        self.covariance = k_decayed_covariance + self.stat.alpha * k_step_update

        return self.covariance, self.mean, self.variance


OUEstimatorState = namedtuple('OUEstimatorState', ['mu', 'log_theta', 'log_sigma'])


class OUEstimator:
    """
    Recursive Ornstein-Uhlenbeck process parameters estimation in exponentially decaying window
    with arbitrary consecutive updates length.
    """

    def __init__(self, alpha):
        """

        Args:
            alpha:  float in [0, 1], decaying window factor.

        Notes:
            alpha ~ 1 / effective_window_length;
            parameters fitted are: Mu, Log_Theta, Log_Sigma, for process: dX = -Theta *(X - Mu) + Sigma * dW;
        """
        self.alpha = alpha
        self.covariance_estimator = Covariance(2, alpha)
        self.residuals_stat = Zscore(1, alpha)
        self.residuals = None
        self.ls_a = None
        self.ls_b = None
        self.mu = None
        self.log_theta = None
        self.log_sigma = None
        self.x_prev = 0.0

    def get_state(self):
        """
        Convenience wrapper.

        Returns:
            current state as instance of OUProcessParams tuple
        """
        return OUEstimatorState(
            mu=self.mu,
            log_theta=self.log_theta,
            log_sigma=self.log_sigma
        )

    def reset(self, trajectory):
        """
        Resets estimator parameters for process dX = -Theta *(X - Mu) + Sigma * dW
        given initial data.

        Args:
            trajectory:     initial 1D process observations trajectory of size [num_points]

        Returns:
            current estimated Mu, Theta, Sigma
        """
        # Fit trajectory:
        x = trajectory[:-1]
        y = trajectory[1:]
        xy = np.stack([x, y], axis=0)

        self.ls_a, self.ls_b = self.fit_ls_estimate(*self.covariance_estimator.reset(xy))

        self.residuals = y - (self.ls_a * x + self.ls_b)

        _, residuals_var = self.residuals_stat.reset(self.residuals[None, :])

        _, self.log_theta, self.log_sigma = self.fit_ou_estimate(self.ls_a, self.ls_b, residuals_var)

        self.mu = self.covariance_estimator.mean.mean()

        self.x_prev = trajectory[-1]

        return self.mu, self.log_theta, self.log_sigma

    def update(self, trajectory, disjoint=False):
        """
        Updates OU parameters values for process dX = -Theta *(X - Mu) + Sigma * dW
        given new observations.

        Args:
            trajectory:  1D process observations trajectory update of size [num_points]
            disjoint:    bool, indicates whether update given is continuous or disjoint w.r.t. previous one

        Returns:
            current estimated  Mu, Theta, Sigma
        """
        if disjoint:
            x = trajectory[:-1]
            y = trajectory[1:]

        else:
            # continuous update, can use backed-up value:
            x = np.concatenate([[self.x_prev], trajectory[:-1]])
            y = trajectory

        xy = np.stack([x, y], axis=0)

        # Fit least squares params:
        self.ls_a, self.ls_b = self.fit_ls_estimate(*self.covariance_estimator.update(xy))

        # Get LS errors and variance:
        self.residuals = y - (self.ls_a * x + self.ls_b)
        _, residuals_var = self.residuals_stat.update(self.residuals[None, :])

        # Get OU params:
        _, self.log_theta, self.log_sigma = self.fit_ou_estimate(self.ls_a, self.ls_b, np.squeeze(residuals_var))
        # Stable mean:
        self.mu = self.covariance_estimator.mean.mean()

        self.x_prev = trajectory[-1]

        return self.mu, self.log_theta, self.log_sigma

    @staticmethod
    def fit_ls_estimate(sigma_xy, mean, variance):
        """
        Computes LS parameters given data covariance matrix, mean and variance: y = a*x + b + e

        Args:
            sigma_xy:   x, y covariance matrix of size [2, 2]
            mean:       x, y mean of size [2]
            variance:   x, y variance of size [2]

        Returns:
            fitted least squares parameters
        """
        a = (sigma_xy / np.clip((variance[0] * variance[1]) ** .5, 1e-6, None))[0, 1]
        b = mean[1] - mean[0] * a

        return np.clip(a, 1e-6, 0.999999), b

    @staticmethod
    def fit_ou_estimate(a, b, err_var, dt=1):
        """
        Given least squares parameters of data and errors variance,
        returns parameters of OU process.

        Args:
            a:          ls slope value
            b:          ls bias value
            err_var:    error variance
            dt:         time increment

        Returns:
            mu, log_theta, log_sigma
        """
        theta = float(- np.log(a) / dt)
        mu = 0.0  # b / (1 - a)  # unstable for a --> 0
        sigma = float((err_var * -2 * np.log(a) / (dt * (1 - a ** 2))) ** .5)
        return mu, np.log(np.clip(theta, 1e-8, None)), np.log(np.clip(sigma, 1e-10, None))


class EMA:
    """
    Recursive exponentially decayed mean estimation for time-series
    with arbitrary consecutive updates length.
    """

    def __init__(self, dim, alpha):
        """

        Args:
            dim:        observation dimensionality
            alpha:      float, decaying factor in [0, 1]

        """
        self.dim = dim
        if alpha is None:
            self.alpha = 1
            self.is_decayed = False
        else:
            self.alpha = alpha
            self.is_decayed = True

        self.mean = None
        self.g = None
        self.num_obs = 0

    def get_state(self):
        """

        Returns:
            current mean value
        """
        return self.mean

    def reset(self, init_x):
        """
        Resets statistics estimates.

        Args:
            init_x:  np.array of initial observations of size [dim, num_init_observations]

        Returns:
            initial dimension-wise mean estimates of sizes [dim, 1]

        """
        if init_x is None:
            self.mean = np.zeros(self.dim)
            self.g = np.zeros(self.dim)
            self.num_obs = 1
            if not self.is_decayed:
                self.alpha = 1

        else:
            assert init_x.ndim == 2 and init_x.shape[0] == self.dim, \
                'Expected init. value as 2D array of size: [{}, num_init_points], got: {}'.format(self.dim,
                                                                                                  init_x.shape)
            self.mean = init_x.mean(axis=-1)
            self.num_obs = init_x.shape[-1]

            if not self.is_decayed:
                self.alpha = 1 / (self.num_obs - 1)

        return self.mean[:, None]

    def update(self, x):
        """
        Updates statistics estimates.

        Args:
            x: np.array, partial trajectory of shape [dim, num_updating_points]

        Returns:
            current dimension-wise mean estimates of size [dim, num_updating_points]
        """
        assert x.ndim == 2 and x.shape[0] == self.dim, \
            'Expected update value as 2D array of size: [{}, num_points], got: {}'.format(self.dim, x.shape)

        # Update length:
        k = x.shape[-1]

        self.num_obs += k
        if not self.is_decayed:
            self.alpha = 1 / (self.num_obs - 1)

        # Mean estimation:

        # Broadcast input to [dim, update_len, update_len]:
        xx = np.tile(x[:, None, :], [1, k, 1])

        gamma = 1 - self.alpha

        # Exp. decays as powers of (1-alpha):
        g = np.cumprod(np.repeat(gamma, k))

        # Diag. matrix of decayed coeff:
        tp = toeplitz(g / gamma, r=np.zeros(k))[::-1, ::1]

        # Backward-ordered mean updates as sums of decayed inputs:
        k_step_mean_update = np.sum(xx * tp[None, ...], axis=2)  # tp expanded for sure broadcast

        # Broadcast stored value of mean to [dim, 1] and apply decay:
        k_decayed_old_mean = (np.tile(self.mean[..., None], [1, k]) * g)

        # Get backward-recursive array of mean values from (num_obs - update_len) to (num_obs):
        means = k_decayed_old_mean + self.alpha * k_step_mean_update[:, ::-1]

        self.g = g

        # Update current value:
        self.mean = means[:, -1]

        return means


STEstimatorState = namedtuple('STEstimatorState', ['df', 'loc', 'scale'])


class STEstimator:
    """
    Standart student-t parameters estimation.
    """
    # TODO: implement true recursive decaying algorithm

    def __init__(self, alpha):
        """

        Args:
            alpha:  float in [0, 1], decaying window factor.

        Notes:
            alpha ~ 1 / effective_window_size;
            parameters fitted are: df, loc, scale  - degree of freedom
        """
        assert alpha is not None and 0 < alpha <= 1.0,\
            "expected alpha as float in [0, 1], got: {}".format(alpha)

        self.alpha = alpha
        # Half-effective tracking window:
        self.window_size = int(np.clip(.5//alpha, 2, None))
        self.trajectory = np.zeros(self.window_size)
        self.mask_idx = 2
        self.df = None
        self.loc = None
        self.scale = None

    def get_state(self):
        """
        Convenience wrapper.

        Returns:
            current state as instance of STEstimatorState tuple
        """
        return STEstimatorState(
            df=self.df,
            loc=self.loc,
            scale=self.scale
        )

    def fit(self, trajectory=None):
        """
        Fits parameters to currently stored or provided data.

        Args:
            trajectory:     array_like, data to fit or None

        Returns:
            fitted parameters: fd, loc, state
        """
        if trajectory is None:
            if self.df is None:
                self.df, self.loc, self.scale = student_t.fit(self.trajectory[-self.mask_idx:])
                self.df = np.clip(self.df, 3, None)

            return self.df, self.loc, self.scale

        else:
            df, loc, scale = student_t.fit(trajectory)
            df = np.clip(df, 3, None)

            return df, loc, scale

    def reset(self, init_trajectory):
        """
        Resets estimator trajectory and parameters with initial data.

        Args:
            init_trajectory:     initial 1D process observations trajectory of size [num_points]
        """
        assert init_trajectory.ndim == 1, 'Expected 1D data, got shape: {}'.format(init_trajectory.shape)
        self.trajectory = np.zeros(self.window_size)
        self.mask_idx = np.clip(len(init_trajectory), None, self.window_size)
        self.trajectory[-self.mask_idx:] = init_trajectory[-self.mask_idx:]

        self.df = None
        self.loc = None
        self.scale = None

    def update(self, trajectory):
        """
        Updates stored trajectory with new observations.

        Args:
            trajectory:  1D process observations trajectory update of size [num_points]

        """
        assert trajectory.ndim == 1, 'Expected 1D data, got shape: {}'.format(trajectory.shape)
        upd_len = np.clip(len(trajectory), None, self.window_size)
        self.mask_idx = np.clip(self.mask_idx + upd_len, None, self.window_size)
        self.trajectory = np.concatenate([self.trajectory[upd_len:], trajectory[-upd_len:]])

        self.df = None
        self.loc = None
        self.scale = None






