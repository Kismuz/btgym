import numpy as np
import copy
from collections import namedtuple

#from profilestats import profile

SSAstate = namedtuple(
    'SSAstate',
    ['window', 'length', 'grouping', 'alpha', 'covariance', 'u', 'singular_values', 'mean', 'variance']
)


class RecSSA:
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
        self.cov_estimator = ExpCovariance(window, alpha=alpha)
        self.covariance = None
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
            self.cov_estimator.stat.mu,
            self.cov_estimator.stat.var,
        )
        return self.state

    def reset(self, x_init=None):
        """..."""
        if x_init is not None:
            assert x_init.shape[0] == self.length
            self.x_embedding = self._delay_embed(np.squeeze(x_init), self.window)
            self.covariance = self.cov_estimator.reset(self.x_embedding)
            self._update_svd()

        else:
            self.x_embedding = np.zeros([self.window, self.length])
            self.covariance = self.cov_estimator.reset()
            self.u = None
            self.singular_values = None
            self.v = None

        return self.x_embedding, self.get_state()

    def update(self, single_x):
        self._update_embed(single_x)
        self.covariance = self.cov_estimator.update(self.x_embedding[:, -1])
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
            B, J, s = RecSSA._henkel_diag_average(d_x.T, n, state.window)
            x_comp.append(np.squeeze(J.T.dot(B) * s))

        return np.asarray(x_comp)


class ExpZscore:
    """
    Recursive exponentially decayed mean and variance estimation.
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

        self.mu = np.zeros(dim)
        self.var = np.zeros(dim)
        self.num_obs = 0

    def reset(self, init_x_matrix=None):
        """
        Resets statistics estimates.

        Args:
            init_x_matrix:  np.array of size[dim, num_init_observations]

        Returns:

        """
        if init_x_matrix is None:
            self.mu = np.zeros(self.dim)
            self.var = np.zeros(self.dim)
            self.num_obs = 0
            if not self.is_decayed:
                self.alpha = 1

        else:
            if self.dim > 1:
                assert init_x_matrix.shape[0] == self.dim

            self.mu = init_x_matrix.mean(axis=-1)
            self.var = init_x_matrix.var(axis=-1)
            self.num_obs = init_x_matrix.shape[-1]
            if not self.is_decayed:
                self.alpha = 1 / (self.num_obs - 1)

        return self.mu, self.var

    def update(self, x):
        """
        Updates current estimates.

        Args:
            x: np.array, single observation of shape [dim, stride]

        Returns:

        """
        assert len(x.shape) == 2 and x.shape[0] == self.dim

        self.num_obs += 1
        if not self.is_decayed:
            self.alpha = 1 / (self.num_obs - 1)

        try:
            dx = x - self.mu[:, None]

        except IndexError:
            dx = x - self.mu[None, None]

        self.var = (1 - self.alpha) * (self.var + self.alpha * (dx ** 2).mean(axis=-1))
        self.mu = (1 - self.alpha) * self.mu + self.alpha * x.mean(axis=-1)

        return self.mu, self.var


class ExpCovariance:
    """
    Recursive exponentially decaying mean, variance and covariance matrix estimation.
    """

    def __init__(self, dim, alpha=None):
        """

        Args:
            dim:        observation dimensionality
            alpha:      float, decaying factor in [0, 1]
        """
        self.stat = ExpZscore(dim, alpha)
        self.covariance = np.eye(dim)

    def reset(self, init_x_matrix=None):
        if init_x_matrix is None:
            _ = self.stat.reset()
            self.covariance = np.eye(self.stat.dim)
            return None

        else:
            _ = self.stat.reset(init_x_matrix)
            self.covariance = np.cov(init_x_matrix)
            return self.covariance

    def update(self, x):
        """
        Updates current estimates.

        Args:
            x: np.array of shape [dim], single observation

        Returns:

        """
        assert x.shape[0] == self.stat.dim
        dx = (x - self.stat.mu)[..., None]
        self.covariance = (1 - self.stat.alpha) * (self.covariance + self.stat.alpha * dx.dot(dx.T))

        _ = self.stat.update(x[:, None])
        return self.covariance

