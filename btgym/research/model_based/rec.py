import numpy as np
from scipy.linalg import toeplitz
import copy
from collections import namedtuple

# from profilestats import profile

SSAstate = namedtuple(
    'SSAstate',
    ['window', 'length', 'grouping', 'alpha', 'covariance', 'u', 'singular_values', 'mean', 'variance']
)


class SSA:
    """
    Recursive toeplitz-style Singular Specter Analysis estimation of 1D signal.
    """

    def __init__(self, window, length, grouping=None, alpha=None):
        """

        Args:
            window:     uint, time embedding window, should be << 'length'
            length:     uint, signal observed length
            grouping:   SSA decomposition triples grouping, iterable of pairs convertible to python slices, i.e.:
                        grouping=[[0,1], [1,2], [2, None]]
            alpha:      float, observation weight decaying factor, in: [0, 1].
        """
        self.window = window
        self.length = length
        self.grouping = grouping
        self.x_embedding = np.zeros([self.window, self.length])
        self.cov_estimator = Covariance(window, alpha=alpha)
        self.covariance = None
        self.mean = None
        self.variance = None
        self.u = None
        self.singular_values = None
        self.v = None
        self.state = None

    def get_state(self):
        self.state = SSAstate(
            self.window,
            self.length,
            self.grouping,
            self.cov_estimator.stat.alpha,
            self.covariance,
            self.u,
            self.singular_values,
            self.mean,
            self.variance,
        )
        return self.state

    def reset(self, x_init=None):
        """..."""
        if x_init is not None:
            assert x_init.shape[0] == self.length
            self.x_embedding = self._delay_embed(np.squeeze(x_init), self.window)
            self.covariance, self.mean, self.variance = self.cov_estimator.reset(self.x_embedding)
            self._update_svd()

        else:
            self.x_embedding = np.zeros([self.window, self.length])
            self.covariance, self.mean, self.variance = self.cov_estimator.reset()
            self.u = None
            self.singular_values = None
            self.v = None

        return self.x_embedding, self.get_state()

    def update(self, single_x):
        self._update_embed(single_x)
        self.covariance, self.mean, self.variance = self.cov_estimator.update(self.x_embedding[:, -1])
        self._update_svd()

        return self.x_embedding, self.get_state()

    def transform(self, x=None, state=None):
        """
        Return SSA signal decomposition.

        Args:
            x:      embedded signal of size [window, length] or None
            state:  instance of SSAstate or None

        Returns:
            SSA signal decomposition of given X w.r.t. given state
            if no arguments provided - returns decomposition of last update;

        """
        if x is None:
            x = self.x_embedding
        else:
            assert state is not None, 'SSAstate is expected when outer X is given, but got: None'

        if state is None:
            state = self.state

        return self._transform(x, state)

    def _update_embed(self, single_x):
        x = np.squeeze(single_x)[None]
        assert x.shape == (1,)
        x_vect = np.concatenate([self.x_embedding[-1, 1:], x])
        self.x_embedding = np.concatenate([self.x_embedding[1:, :], x_vect[None, ...]], axis=0)

    def _update_svd(self):
        """
        Toeplitz variant of SSA decomposition (based on covariance matrix).
        """
        self.u, self.singular_values, self.v = np.linalg.svd(self.covariance)

    @staticmethod
    def _delay_embed(x, w):
        """
        Time-embedding with window size `w` and stride 1
        """
        g = 0
        return x[(np.arange(w ) * (g + 1)) + np.arange(np.max(x.shape[0] - (w - 1) * (g + 1), 0)).reshape(-1, 1)].T

    # @staticmethod
    # # @profile(print_stats=1)
    # def ___henkel_diag_average(x, n, window):
    #     """
    #     Computes  diagonal averaging operator D.
    #     Usage: D = J.T.dot(B)*s, see:
    #     Dimitrios D. Thomakos, `Optimal Linear Filtering, Smoothing and Trend Extraction
    #     for Processes with Unit Roots and Cointegration`, 2008; pt. 2.2
    #     """
    #     J = np.ones([n - window + 1, 1])
    #     pad = n - window
    #     B = np.asarray(
    #         [
    #             np.pad(x[i, :], [i, pad - i], mode='constant', constant_values=[np.nan, np.nan])
    #             for i in range(x.shape[0])
    #         ]
    #     )
    #     B = np.ma.masked_array(B, mask=np.isnan(B))
    #     s = 1 / np.logical_not(B.mask).sum(axis=0)
    #     B[B.mask] = 0.0
    #     return B.data, J, s

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
        assert isinstance(state, SSAstate)
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


class Zscore1:
    """
    DEPRECATED, USE Zscore instead

    Recursive exponentially decayed mean and variance estimation with single point update.
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

        self.mean = np.zeros(dim)
        self.variance = np.zeros(dim)
        self.num_obs = 0

    def reset(self, init_x=None):
        """
        Resets statistics estimates.

        Args:
            init_x:  np.array of size[dim, num_init_observations]

        Returns:

        """
        if init_x is None:
            self.mean = np.zeros(self.dim)
            self.variance = np.zeros(self.dim)
            self.num_obs = 1
            if not self.is_decayed:
                self.alpha = 1

        else:
            if self.dim > 1:
                assert init_x.shape[0] == self.dim

            self.mean = init_x.mean(axis=-1)
            self.variance = init_x.var(axis=-1)
            self.num_obs = init_x.shape[-1]
            if not self.is_decayed:
                self.alpha = 1 / (self.num_obs - 1)

        return self.mean, self.variance

    def update(self, x):
        """
        Updates current estimates.

        Args:
            x: np.array, single observation of shape [dim, 1]

        Returns:
            updated mean and variance of sizes [dim, 1]
        """
        assert len(x.shape) == 2 and x.shape[0] == self.dim

        self.num_obs += 1
        if not self.is_decayed:
            self.alpha = 1 / (self.num_obs - 1)

        try:
            dx = x - self.mean[:, None]

        except IndexError:
            dx = x - self.mean[None, None]

        self.variance = (1 - self.alpha) * (self.variance + self.alpha * (dx ** 2).mean(axis=-1))
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x.mean(axis=-1)

        return self.mean, self.variance


class Covariance1:
    """
    DEPRECATED, USE Covariance instead
    Recursive exponentially decaying mean, variance and covariance matrix estimation.
    """

    def __init__(self, dim, alpha=None):
        """

        Args:
            dim:        observation dimensionality
            alpha:      float, decaying factor in [0, 1]
        """
        self.stat = Zscore1(dim, alpha)
        self.covariance = np.eye(dim)
        self.mean = None
        self.variance = None

    def reset(self, init_x=None):
        if init_x is None:
            self.mean, self.variance = self.stat.reset()
            self.covariance = np.eye(self.stat.dim)

        else:
            self.mean, self.variance = self.stat.reset(init_x)
            self.covariance = np.cov(init_x)

        return self.covariance, self.mean, self.variance

    def update(self, x):
        """
        Updates current estimates.

        Args:
            x: np.array of shape [dim, 1], single observation

        Returns:

        """
        assert x.shape[0] == self.stat.dim
        dx = x - self.stat.mean[..., None]
        self.covariance = (1 - self.stat.alpha) * (self.covariance + self.stat.alpha * dx.dot(dx.T))

        self.mean, self.variance = self.stat.update(x)
        return self.covariance, self.mean, self.variance


class Zscore:
    """
    Recursive exponentially decayed mean and variance estimation  for time-series
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

        self.mean = np.zeros(dim)
        self.variance = np.zeros(dim)
        self.g = None
        self.dx = None
        self.num_obs = 0

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
            self.variance = np.zeros(self.dim)
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

        # Exp. decays of (1-alpha):
        g = np.cumprod(np.repeat(gamma, k))

        # Diag. matrix of decayed coeff:
        tp = toeplitz(g / gamma, r=np.zeros(k))[::-1, ::1]

        # Sums of decayed inputs:
        sum_x = np.sum(xx * tp[None, ...], axis=2)  # tp expanded for shure broadcast

        # Broadcast stored value of mean to [dim, 1] and apply decay:
        decayed_old_mean = (np.tile(self.mean[..., None], [1, k]) * g)

        # Get backward-recursive array of mean values from (num_obs - update_len) to (num_obs):
        mean = decayed_old_mean + self.alpha * sum_x[:, ::-1]

        # Variance estimation:

        # Get deviations of update:
        dx = x - np.concatenate([self.mean[..., None], mean[:, :-1]], axis=1)

        # Get new variance value at (num_obs) point:
        decayed_old_var = gamma ** k * self.variance
        k_step_update = np.sum(g[::-1] * dx ** 2, axis=1)

        variance = decayed_old_var + self.alpha * k_step_update

        # Update current values:
        self.mean = mean[:, -1]
        self.variance = variance

        # Keep g and dx:
        self.g = g
        self.dx = dx

        return self.mean, self.variance


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
        self.mean = None
        self.variance = None

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


class OUEstimator:
    """
    Recursive Ornshtein-Uhlenbeck process parameters estimation in exponentially decaying window.
    """

    def __init__(self, alpha):
        """

        Args:
            alpha:      float, decaying factor in [0, 1]
        """
        self.alpha = alpha
        self.covariance_estimator = Covariance(2, alpha)
        self.error_stat = Zscore(1, alpha)
        self.ls_a = 0.0
        self.ls_b = 0.0
        self.mu = 0.0
        self.l = 0.0
        self.sigma = 0.0
        self.x_prev = 0.0

    def reset(self, trajectory=None):
        """

        Args:
            trajectory:     initial 1D process observations trajectory of size [num_points] or None

        Returns:
            current estimated OU mu, lambda, sigma
        """
        # TODO: require init. trajectory
        if trajectory is None:
            self.ls_a = 0.0
            self.ls_b = 0.0
            self.mu = 0.0
            self.mu = 0.0
            self.l = 0.0
            self.sigma = 0.0
            self.x_prev = 0.0
            self.covariance_estimator.reset(None)
            self.error_stat.reset(None)

        else:
            # Fit trajectory:
            x = trajectory[:-1]
            y = trajectory[1:]
            xy = np.stack([x,y], axis=0)

            self.ls_a, self.ls_b = self.fit_ls_estimate(*self.covariance_estimator.reset(xy))

            err = y - (self.ls_a * x + self.ls_b)

            _, err_var = self.error_stat.reset(err)

            _, self.l, self.sigma = self.fit_ou_estimate(self.ls_a, self.ls_b, err_var)

            self.mu = self.covariance_estimator.mean.mean()

            self.x_prev = trajectory[-1]

        return self.mu, self.l, self.sigma

    def update(self, x):
        # TODO: allow arbitrary update length
        """
        Updates OU parameters estimates given single new observation.
        Args:
            x:  single observation

        Returns:
            current estimated OU mu, lambda, sigma
        """
        xy = np.asarray([self.x_prev, np.squeeze(x)])

        self.ls_a, self.ls_b = self.fit_ls_estimate(*self.covariance_estimator.update(xy))

        err = xy[1] - (self.ls_a * xy[0] + self.ls_b)

        _, err_var = self.error_stat.update(np.asarray(err)[None, None])

        _, self.l, self.sigma = self.fit_ou_estimate(self.ls_a, self.ls_b, np.squeeze(err_var))

        # Stable:
        self.mu = self.covariance_estimator.mean.mean()

        self.x_prev = x

        return self.mu, self.l, self.sigma

    @staticmethod
    def fit_ls_estimate(sigma_xy, mean, variance):
        """
        Computes LS parameters given data covariance matrix, mean and variance: y = a*x + b + e

        Args:
            sigma_xy:   x, y covariance matrix of size [2, 2]
            mean:       x, y mean of size [2]
            variance:   x, y variance of size [2]

        Returns:
            estimated least squares parameters
        """
        # a = sigma_xy / np.clip(variance[0], 1e-8, None)
        a = (sigma_xy / np.clip((variance[0] * variance[1])**.5, 1e-6, None))[0, 1]
        # a = (a / np.clip(np.diag(a).mean(), 1e-10, None))[0, 1]
        b = mean[1] - mean[0] * a

        return np.clip(a, 1e-6, 0.999999), b

    @staticmethod
    def fit_ou_estimate(a, b, err_var, dt=1):
        """
        Given least squares parameters of data and error variance,
        returns parameters of OU process.

        Args:
            a:          ls slope value
            b:          ls bias value
            err_var:    error variance
            dt:         time increment

        Returns:
            mu, lambda, sigma
        """
        l = - np.log(a) / dt
        mu = 0.0 #b / (1 - a)
        sigma = (err_var * -2 * np.log(a) / (dt * (1 - a ** 2))) ** .5

        return mu, l, sigma






