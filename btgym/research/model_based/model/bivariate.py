import numpy as np
from collections import namedtuple

from btgym.research.model_based.model.rec import Zscore, ZscoreState, Covariance, CovarianceState
from btgym.research.model_based.model.rec import OUEstimatorState

from btgym.research.model_based.model.univariate import OUProcess, TimeSeriesModel


BivariateTSModelState = namedtuple('BivariateTSModelState', ['p', 's', 'stat', 'ps_stat'])


class BivariateTSModel:
    """
    Two-factor bivariate time-series model.

    Motivating papers:
        Eduardo Schwartz, James E. Smith, "Short-Term Variations and Long-Term Dynamics in Commodity Prices",
        in "Management Science", Vol. 46, No. 7, July 2000 pp. 893–911

        Harris, D., "Principal components analysis of cointegrated time series," in "Econometric Theory", Vol. 13, 1997
    """
    # TODO: trajectory generator uses simplified algorithm: entire trajectory is generated out of single model state
    # TODO: proper state-space model approach
    # TODO: should be: sample [randomized?] trajectory of states -> sample realisation trajectory of same length
    # Decomposition matrix:
    u_decomp = np.asarray([[.5, .5], [.5, -.5]])

    # Reconstruction (inverse u_decomp):
    u_recon = np.asarray([[1.,  1.], [1., -1.]])

    def __init__(
            self,
            max_length,
            analyzer_window,
            p_analyzer_grouping=None,
            s_analyzer_grouping=None,
            alpha=None,
            filter_alpha=None,
            stat_alpha=None,
            ps_alpha=None,
    ):
        """

        Args:
            max_length:             uint, maximum time-series trajectory length to keep;
            analyzer_window:        uint, SSA embedding window (shared for P and S analyzers);
            p_analyzer_grouping:    P process SSA decomposition triples grouping,
                                    iterable of pairs convertible to python slices, i.e.:
                                    grouping=[[0,1], [1,2], [2, None]];
            s_analyzer_grouping:    P process SSA decomposition triples grouping, se above;
            alpha:                  float in [0, 1], SSA and processes estimators decaying factor;
            filter_alpha:           float in [0, 1], processes smoothing decaying factor;
            stat_alpha:             float in [0, 1], time-series statistics tracking decaying factor;
            ps_alpha:               float in [0, 1], P|S processes covariance tracking decaying factor;
        """
        max_length = np.atleast_1d(max_length)
        analyzer_window = np.atleast_1d(analyzer_window)
        alpha = np.atleast_1d(alpha)
        filter_alpha = np.atleast_1d(filter_alpha)

        # Max. variance factor component (average):
        self.p = TimeSeriesModel(
            max_length[0],
            analyzer_window[0],
            p_analyzer_grouping,
            alpha[0],
            filter_alpha[0]
        )

        # Max. stationarity factor component (difference):
        self.s = TimeSeriesModel(
            max_length[-1],
            analyzer_window[-1],
            s_analyzer_grouping,
            alpha[-1],
            filter_alpha[-1]
        )

        # Statistics of original data:
        self.stat = Zscore(2, stat_alpha)

        # Stochastic processes covariance:
        self.ps_stat = Covariance(2, ps_alpha)

    def ready(self):
        return self.s.ready() and self.p.ready()

    def get_state(self):
        return BivariateTSModelState(
            p=self.p.get_state(),
            s=self.s.get_state(),
            stat=self.stat.get_state(),
            ps_stat=self.ps_stat.get_state()
        )

    @staticmethod
    def get_random_state(p_params, s_params, mean=(100, 100), variance=(1, 1), ps_corrcoef=(-1, 1)):
        """
        Samples random uniform model state w.r.t. parameters intervals given.

        Args:
            p_params:       dict, P stochastic process parameters, see kwargs at: OUProcess.get_random_state
            s_params:       dict, S stochastic process parameters, see kwargs at: OUProcess.get_random_state
            mean:           iterable of floats as [lower_bound, upper_bound], time-series means sampling interval.
            variance:       iterable of floats as [lower_bound, upper_bound], time-series variances sampling interval.
            ps_corrcoef:    iterable of floats as [lower_bound, upper_bound], correlation coefficient
                            for P and S process innovations, -1 <= ps_corrcoef <= 1

        Returns:
            instance of BivariateTSModelState

        Note:
            negative means are allowed.
        """
        sample = dict()
        for name, param, low_threshold in zip(
                ['mean', 'variance', 'ps_corrcoef'], [mean, variance, ps_corrcoef], [-np.inf, 1e-8, -1.0]):
            interval = np.asarray(param)
            assert interval.ndim == 1 and interval[0] <= interval[-1], \
                ' Expected param `{}` as iterable of ordered values as: [lower_bound, upper_bound], got: {}'.format(
                    name, interval
                )
            assert interval[0] >= low_threshold, \
                'Expected param `{}` lower bound be no less than {}, got: {}'.format(name, low_threshold, interval[0])

            sample[name] = np.random.uniform(low=interval[0], high=interval[-1], size=2)

        # Correlation matrix instead of covariance - it is ok as it gets normalized when sampling anyway:
        rho = np.eye(2)
        rho[0, 1] = rho[1, 0] = sample['ps_corrcoef'][0]
        # TODO: log-uniform sampling for s, p params
        return BivariateTSModelState(
            p=TimeSeriesModel.get_random_state(**p_params),
            s=TimeSeriesModel.get_random_state(**s_params),
            stat=ZscoreState(
                mean=sample['mean'],
                variance=sample['variance']
            ),
            ps_stat=CovarianceState(
                mean=np.zeros(2),
                variance=np.ones(2),
                covariance=rho,
            ),
        )

    @staticmethod
    def _decompose(trajectory, mean, variance, u):
        """
        Returns orthonormal decomposition of pair [X1, X2].
        Static method, can be used as stand-along function.

        Args:
            trajectory: time-series data of shape [2, num_points]
            mean:       data mean of size [2]
            variance:   data variance of size [2]
            u:          [2, 2] decomposition matrix

        Returns:
            data projection of size [2, num_pints], where first (P) component is average and second (S) is difference
            of original time-series.
        """
        assert len(trajectory.shape) == 2 and trajectory.shape[0] == 2, \
            'Expected data as array of size [2, num_points], got: {}'.format(trajectory.shape)

        assert mean.shape == (2,) and variance.shape == (2,), \
            'Expected mean and variance as vectors of size [2], got: {}, {}'.format(mean.shape, variance.shape)

        assert u.shape == (2, 2), 'Expected U as 2x2 matrix, got: {}'.format(u.shape)

        # Z-score data:
        # Mind swapped STD!
        norm_data = (trajectory - mean[:, None]) / np.clip(variance[:, None], 1e-8, None) ** .5
        ps_decomposition = np.matmul(u, norm_data)

        return ps_decomposition

    @staticmethod
    def _reconstruct(ps_decomposition, mean, variance, u):
        """
        Returns original data [X1, X2] given orthonormal P|S decomposition .
        Static method, can be used as stand-along function.

        Args:
            ps_decomposition:   data ps-decomposition of size [2, num_points]
            mean:               original data mean of size [2]
            variance:           original data variance of size [2]
            u:                  [2, 2] reconstruction matrix

        Returns:
            reconstructed data of size [2, num_pints]
        """
        assert len(ps_decomposition.shape) == 2 and ps_decomposition.shape[0] == 2, \
            'Expected data as array of size [2, num_points], got: {}'.format(ps_decomposition.shape)

        assert mean.shape == (2,) and variance.shape == (2,), \
            'Expected mean and variance as vectors of size [2], got: {}, {}'.format(mean.shape, variance.shape)

        assert u.shape == (2, 2), 'Expected U as 2x2 matrix, got: {}'.format(u.shape)

        return np.matmul(u, ps_decomposition) * variance[:, None] ** .5 + mean[:, None]

    def decompose(self, trajectory):
        """
        Returns orthonormal decomposition of pair [X1, X2] w.r.t current statistics.

        Args:
            trajectory: time-series data of shape [2, num_points]

        Returns:
            tuple (P, S), where first (P) component is average and second (S) is difference
            of original time-series, of size [num_points] each
        """
        ps_decomp = self._decompose(trajectory, self.stat.mean, self.stat.variance, self.u_decomp)
        return ps_decomp[0, :], ps_decomp[1, :]

    def reconstruct(self, p, s, mean=None, variance=None):
        """
        Returns original data [X1, X2] given orthonormal P|S decomposition.

        Args:
            p:          data p-component of shape [num_points]
            s:          data s-component of shape [num_points]
            mean:       original data mean of size [2] or None
            variance:   original data variance of size [2] or None

        Returns:
            reconstructed data of size [2, num_pints]

        Notes:
            if either mean or variance arg is not given - stored mean and variance are used.
        """
        assert p.shape == s.shape, ' Expected components be same size but got: {} and {}'.format(p.shape, s.shape)

        if mean is None or variance is None:
            mean = self.stat.mean
            variance = self.stat.variance

        ps = np.stack([p, s], axis=0)
        return self._reconstruct(ps, mean, variance, self.u_recon)

    def reset(self, init_trajectory):
        """
        Resets model parameters and trajectories given initial data.

        Args:
            init_trajectory:    initial time-series observations of size [2, num_points]
        """
        _ = self.stat.reset(init_trajectory)
        p_data, s_data = self.decompose(init_trajectory)
        self.p.reset(p_data)
        self.s.reset(s_data)
        residuals = np.stack(
            [self.p.process.estimator.residuals, self.s.process.estimator.residuals],
            axis=0
        )
        _ = self.ps_stat.reset(residuals)

    def update(self, trajectory, disjoint=False):
        """
        Updates model parameters and trajectories given new data.

        Args:
            trajectory: time-series update observations of size [2, num_points], where:
                        num_points <= min{p_params[max_length], s_params[max_length]} is necessary
                        to keep model trajectory continuous
            disjoint:   bool, indicates whether update given is continuous or disjoint w.r.t. previous one
        """
        _ = self.stat.update(trajectory)  # todo: this stat.estimator does not respect `disjoint` arg.; ?!!
        p_data, s_data = self.decompose(trajectory)
        self.p.update(p_data, disjoint)
        self.s.update(s_data, disjoint)
        residuals = np.stack(
            [self.p.process.estimator.residuals, self.s.process.estimator.residuals],
            axis=0
        )
        _ = self.ps_stat.update(residuals)

    def transform(self, trajectory=None, state=None, size=None):
        """
        Returns per-component analyzer data decomposition.

        Args:
            trajectory:     bivariate data to decompose of size [2, num_points] or None
            state:          instance of BivariateTSModelState or None
            size:           uint, size of decomposition to get, or None

        Returns:
            array of [size or num_points], array of [size or num_points], ZscoreState(2)

            - SSA transformations of P, S components of given trajectory w.r.t. given state
            - bivariate trajectory statistics (means and variances)

        Notes:
            if no `trajectory` is given - returns stored data decomposition
            if no `state` arg. is given - uses stored analyzer state.
            if no 'size` arg is given - decomposes full [stored or given] trajectory
        """
        if state is not None:
            assert isinstance(state, BivariateTSModelState),\
                'Expected `state as instance of BivariateTSModelState but got: {}`'.format(type(state))
            s_state = state.s
            p_state = state.p
            stat = state.stat

        else:
            assert trajectory is None, 'When `trajectory` arg. is given, `state` is required'
            p_state = None
            s_state = None
            stat = self.stat.get_state()

        if trajectory is not None:
            ps_data = self._decompose(trajectory, stat.mean, stat.variance, self.u_decomp)
            p_data = ps_data[0, :]
            s_data = ps_data[1, :]

        else:
            p_data = None
            s_data = None

        p_transformed = self.p.transform(p_data, p_state, size)
        s_transformed = self.s.transform(s_data, s_state, size)

        return p_transformed, s_transformed, stat

    def get_trajectory(self, size=None, reconstruct=True):
        """
        Returns stored decomposition fragment and [optionally] time-series reconstruction.
        TODO: reconstruction is freaky due to only last stored statistic is used

        Args:
            size:           uint, fragment length to get in [1, ..., max_length] or None
            reconstruct:    bool, if True - also return data reconstruction

        Returns:
            array of [size ... max_length], array of [size ... max_length], array of size [2, size ... max_length]
            or
            array of [size ... max_length], array of [size ... max_length], None

            P,C [, and 2D trajectory] series as [ x[-size], x[-size+1], ... x[-1] ], up to length [size];
            if no `size` arg. is given - returns entire stored trajectory, up to length [max_length].

        """
        p_data = self.p.get_trajectory(size)
        s_data = self.s.get_trajectory(size)

        if reconstruct:
            trajectory = self.reconstruct(p_data, s_data)

        else:
            trajectory = None

        return p_data, s_data, trajectory

    @staticmethod
    def generate_trajectory_fn(batch_size, size, state, reconstruct=False, u_recon=None):
        """
        Generates batch of time-series realisations given model state.
        Static method, can be used as stand-along function.

        Args:
            batch_size:     uint, number of trajectories to generates
            size:           uint, trajectory length to generate
            state:          instance of BivariateTSModelState;
            reconstruct:    bool, if True - return time-series along with P, S trajectories, return None otherwise
            u_recon:        reconstruction matrix of size [2, 2] or None; required if reconstruct=True;

        Returns:
            generated P and S processes realisations of size [batch_size, 2, size];
            generated time-series reconstructions of size [batch_size, 2, size] or None;
        """
        assert isinstance(state, BivariateTSModelState), \
            'Expected `state` as instance of BivariateTSModelState, got: {}'.format(type(state))

        if reconstruct:
            assert u_recon is not None, 'reconstruct=True but reconstruction matrix is not provided.'

        # Unpack:
        p_state = state.p.process
        s_state = state.s.process

        # Get all samples for single batch (faster):
        p_params = OUProcess.sample_naive_unbiased(p_state, batch_size)
        s_params = OUProcess.sample_naive_unbiased(s_state, batch_size)

        # Concatenate batch-wise:
        parameters = OUEstimatorState(
            mu=np.concatenate([p_params.mu, s_params.mu]),
            log_theta=np.concatenate([p_params.log_theta, s_params.log_theta]),
            log_sigma=np.concatenate([p_params.log_sigma, s_params.log_sigma]),
        )
        driver_df = np.concatenate(
            [
                np.tile(p_state.driver_df, batch_size),
                np.tile(s_state.driver_df, batch_size),
            ]
        )
        # Access multivariate generator_fn directly to get batch of bivariate OU:
        batch_2x = OUProcess.generate_trajectory_fn(2 * batch_size, size, parameters, driver_df)
        batch_2x = np.reshape(batch_2x, [2, batch_size, -1])
        batch_2x = np.swapaxes(batch_2x, 0, 1)

        if reconstruct:
            x = np.matmul(u_recon, batch_2x) * state.stat.variance[None, :, None] ** .5 \
                + state.stat.mean[None, :, None]

        else:
            x = None

        return batch_2x, x

    @staticmethod
    def generate_bivariate_trajectory_fn(batch_size, size, state, reconstruct=False, u_recon=None):
        """
        Generates batch of time-series realisations given model state.
        Static method, can be used as stand-along function.

        Args:
            batch_size:     uint, number of trajectories to generates
            size:           uint, trajectory length to generate
            state:          instance of BivariateTSModelState;
            reconstruct:    bool, if True - return time-series along with P, S trajectories, return None otherwise
            u_recon:        reconstruction matrix of size [2, 2] or None; required if reconstruct=True;

        Returns:
            generated P and S processes realisations of size [batch_size, 2, size];
            generated time-series reconstructions of size [batch_size, 2, size] or None;
        """
        assert isinstance(state, BivariateTSModelState), \
            'Expected `state` as instance of BivariateTSModelState, got: {}'.format(type(state))

        if reconstruct:
            assert u_recon is not None, 'reconstruct=True but reconstruction matrix is not provided.'

        # Unpack:
        p_state = state.p.process
        s_state = state.s.process

        # Get all samples for single batch (faster):
        p_params = OUProcess.sample_naive_unbiased(p_state, 1)
        s_params = OUProcess.sample_naive_unbiased(s_state, 1)

        # Concatenate batch-wise:
        parameters = OUEstimatorState(
            mu=np.concatenate([p_params.mu, s_params.mu]),
            log_theta=np.concatenate([p_params.log_theta, s_params.log_theta]),
            log_sigma=np.concatenate([p_params.log_sigma, s_params.log_sigma]),
        )
        driver_df = np.asarray([p_state.driver_df, s_state.driver_df])

        # Access multivariate generator_fn directly to get batch of 2d correlated OU's:
        batch_2d = OUProcess.generate_multivariate_trajectory_fn(
            batch_size=batch_size,
            size=size,
            parameters=parameters,
            t_df=driver_df,
            covariance=state.ps_stat.covariance
        )
        batch_2d = np.swapaxes(batch_2d, 1, 2)

        if reconstruct:
            x = np.matmul(u_recon, batch_2d) * state.stat.variance[None, :, None] ** .5 \
                + state.stat.mean[None, :, None]

        else:
            x = None

        return batch_2d, x

    def generate(self, batch_size, size, state=None, reconstruct=True):
        """
        Generates batch of time-series realisations given model state.

        Args:
            batch_size:     uint, number of trajectories to generates
            size:           uint, trajectory length to generate
            state:          instance of BivariateTSModelState or None;
                            if no state provided - current state is used.
            reconstruct:    bool, if True - return time-series along with P, S trajectories, return None otherwise

        Returns:
            generated P and S processes realisations of size [batch_size, 2, size];
            generated time-series reconstructions of size [batch_size, 2, size] or None;
        """
        if state is None:
            # Fit student-t df:
            _ = self.p.process.driver_estimator.fit()
            _ = self.s.process.driver_estimator.fit()

            state = self.get_state()

        # return self.generate_trajectory_fn(batch_size, size, state, reconstruct, self.u_recon)
        return self.generate_bivariate_trajectory_fn(batch_size, size, state, reconstruct, self.u_recon)


class BivariatePriceModel(BivariateTSModel):
    """
    Wrapper class for positive-valued time-series.
    Internally works with log-transformed data.
    """

    def reset(self, init_trajectory):
        """
        Resets model parameters and trajectories given initial data.

        Args:
            init_trajectory:    initial time-series observations of size [2, num_points]
        """
        return super().reset(np.log(init_trajectory))

    def update(self, trajectory, disjoint=False):
        """
        Updates model parameters and trajectories given new data.

        Args:
            trajectory: time-series update observations of size [2, num_points], where:
                        num_points <= min{p_params[max_length], s_params[max_length]} is necessary
                        to keep model trajectory continuous
            disjoint:   bool, indicates whether update given is continuous or disjoint w.r.t. previous one
        """
        return super().update(np.log(trajectory), disjoint)

    def transform(self, trajectory=None, state=None, size=None):
        """
        Returns per-component analyzer data decomposition.

        Args:
            trajectory:     data to decompose of size [2, num_points] or None
            state:          instance of BivariateTSModelState or None
            size:           uint, size of decomposition to get, or None

        Returns:
            array of [size or num_points], array of [size or num_points], ZscoreState(2)

            - SSA transformations of P, S components of given trajectory w.r.t. given state
            - bivariate trajectory statistics (means and variances)

        Notes:
            if no `trajectory` is given - returns stored data decomposition
            if no `state` arg. is given - uses stored analyzer state.
            if no 'size` arg is given - decomposes full [stored or given] trajectory
        """
        if trajectory is not None:
            trajectory = np.log(trajectory)

        return super().transform(trajectory, state, size)

    def get_trajectory(self, size=None, reconstruct=True):
        """
        Returns stored decomposition fragment and [optionally] time-series reconstruction.
        TODO: reconstruction is freaky due to only last stored statistic is used

        Args:
            size:           uint, fragment length to get in [1, ..., max_length] or None
            reconstruct:    bool, if True - also return data reconstruction

        Returns:
            array of [size ... max_length], array of [size ... max_length], array of size [2, size ... max_length]
            or
            array of [size ... max_length], array of [size ... max_length], None

            P,C [, and 2D trajectory] series as [ x[-size], x[-size+1], ... x[-1] ], up to length [size];
            if no `size` arg. is given - returns entire stored trajectory, up to length [max_length].

        """
        p_data, s_data, trajectory = super().get_trajectory(size, reconstruct)

        if reconstruct:
            trajectory = np.exp(trajectory)

        return p_data, s_data, trajectory

    @staticmethod
    def get_random_state(p_params, s_params, mean=(100, 100), variance=(1, 1), ps_corrcoef=(-1, 1)):
        """
        Samples random uniform model state w.r.t. intervals given.

        Args:
            p_params:       dict, P stochastic process parameters, see kwargs at: OUProcess.get_random_state
            s_params:       dict, S stochastic process parameters, see kwargs at: OUProcess.get_random_state
            mean:           iterable of floats as [0 < lower_bound, upper_bound], time-series means sampling interval.
            variance:       iterable of floats as [0 < lower_bound, upper_bound], time-series variances sampling interval.
            ps_corrcoef:    iterable of floats as [-1 <= lower_bound, upper_bound <= 1], correlation coefficient
                            for P and S process innovations.

        Returns:
            instance of BivariateTSModelState

        Note:
            negative means are rejected;
            P and S processes fitted over log_transformed data;
        """
        sample = dict()
        for name, param, low_threshold in zip(
                ['mean', 'variance', 'ps_corrcoef'], [mean, variance, ps_corrcoef], [1e-8, 1e-8, -1.0]):
            interval = np.asarray(param)
            assert interval.ndim == 1 and interval[0] <= interval[-1], \
                ' Expected param `{}` as iterable of ordered values as: [lower_bound, upper_bound], got: {}'.format(
                    name, interval
                )
            assert interval[0] >= low_threshold, \
                'Expected param `{}` lower bound be no less than {}, got: {}'.format(name, low_threshold, interval[0])

            sample[name] = np.random.uniform(low=interval[0], high=interval[-1], size=2)

        # Correlation matrix instead of covariance - it is ok as it gets normalized when sampling anyway:
        rho = np.eye(2)
        rho[0, 1] = rho[1, 0] = sample['ps_corrcoef'][0]

        # Log_transform mean and variance (those is biased estimates but ok for rnd. samples):
        log_variance = np.log(sample['variance'] / sample['mean'] ** 2 + 1)
        log_mean = np.log(sample['mean']) - .5 * log_variance

        # Inverse transform memo:
        # mean = exp(log_mean + 0.5 * log_var)
        # var = mean**2 * (exp(log_var) -1)

        return BivariateTSModelState(
            p=TimeSeriesModel.get_random_state(**p_params),
            s=TimeSeriesModel.get_random_state(**s_params),
            stat=ZscoreState(
                mean=log_mean,
                variance=log_variance
            ),
            ps_stat=CovarianceState(
                mean=np.zeros(2),
                variance=np.ones(2),
                covariance=rho,
            ),
        )

    @staticmethod
    def generate_trajectory_fn(batch_size, size, state, reconstruct=False, u_recon=None):
        """
        Generates batch of time-series realisations given model state.
        Static method, can be used as stand-along function.

        Args:
            batch_size:     uint, number of trajectories to generates
            size:           uint, trajectory length to generate
            state:          instance of BivariateTSModelState;
            reconstruct:    bool, if True - return time-series along with P, S trajectories, return None otherwise
            u_recon:        reconstruction matrix of size [2, 2] or None; required if reconstruct=True;

        Returns:
            generated P and S processes realisations of size [batch_size, 2, size];
            generated time-series reconstructions of size [batch_size, 2, size] or None;
        """
        batch_2x, x = BivariateTSModel.generate_trajectory_fn(batch_size, size, state, reconstruct, u_recon)

        if reconstruct:
            x = np.exp(x)

        return batch_2x, x

    @staticmethod
    def generate_bivariate_trajectory_fn(batch_size, size, state, reconstruct=False, u_recon=None):
        """
        Generates batch of time-series realisations given model state.
        Static method, can be used as stand-along function.

        Args:
            batch_size:     uint, number of trajectories to generates
            size:           uint, trajectory length to generate
            state:          instance of BivariateTSModelState;
            reconstruct:    bool, if True - return time-series along with P, S trajectories, return None otherwise
            u_recon:        reconstruction matrix of size [2, 2] or None; required if reconstruct=True;

        Returns:
            generated P and S processes realisations of size [batch_size, 2, size];
            generated time-series reconstructions of size [batch_size, 2, size] or None;
        """
        batch_2d, x = BivariateTSModel.generate_bivariate_trajectory_fn(batch_size, size, state, reconstruct, u_recon)

        if reconstruct:
            x = np.exp(x)

        return batch_2d, x


class BPM(BivariatePriceModel):
    """
    Wrapper class with de-facto disabled analyzer
    in favor to state lightness an computation speed.
    """

    def __init__(
            self,
            *args,
            analyzer_window=None,
            p_analyzer_grouping=None,
            s_analyzer_grouping=None,
            **kwargs
    ):
        super().__init__(
            *args,
            analyzer_window=[2, 2],
            p_analyzer_grouping=None,
            s_analyzer_grouping=None,
            **kwargs
        )






