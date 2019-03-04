import numpy as np
from scipy import stats
from collections import namedtuple

from btgym.research.model_based.model.stochastic import ou_process_t_driver_batch_fn
from btgym.research.model_based.model.stochastic import multivariate_ou_process_t_driver_batch_fn

from btgym.research.model_based.model.rec import Zscore, ZscoreState, Covariance, CovarianceState
from btgym.research.model_based.model.rec import SSA, OUEstimator, OUEstimatorState, STEstimator

OUProcessState = namedtuple('OUProcessState', ['observation', 'filtered', 'driver_df'])


class OUProcess:
    """
    Provides essential functionality for recursive time series modeling
    as Ornshteinh-Uhlenbeck stochastic process:
    parameters estimation, state filtering and sampling, trajectories generation.
    """
    def __init__(self, alpha=None, filter_alpha=None):
        self.alpha = alpha
        self.filter_alpha = filter_alpha
        self.estimator = OUEstimator(alpha)

        # Just use exponential smoothing as state-space trajectory filter:
        self.filter = Covariance(3, alpha=filter_alpha)

        # Driver is Student-t:
        self.driver_estimator = STEstimator(alpha)

        # Empirical statistics tracker (debug, mostly for accuracy checking, not included in OUProcessState):
        self.stat = Zscore(1, alpha)

        self.is_ready = False

    def ready(self):
        assert self.is_ready, 'OUProcess is not initialized. Hint: forgot to call .reset()?'

    def get_state(self):
        """
        Returns model state tuple.

        Returns:
            current state as instance of OUProcessState
        """
        self.ready()
        return OUProcessState(
            observation=self.estimator.get_state(),
            filtered=self.filter.get_state(),
            driver_df=self.driver_estimator.df,
        )

    @staticmethod
    def get_random_state(mu=(0, 0), theta=(.1, 1), sigma=(0.1, 1), driver_df=(3, 50), variance=1e-2):
        """
        Samples random uniform process state w.r.t. parameters intervals given.

        Args:
            mu:         iterable of floats as [lower_bound, upper_bound], OU Mu sampling interval
            theta:      iterable of positive floats as [lower_bound, upper_bound], OU Theta sampling interval
            sigma:      iterable of positive floats as [lower_bound, upper_bound], OU Sigma sampling interval
            driver_df:  iterable of positive floats as [lower_bound > 2, upper_bound],
                        student-t driver degrees of freedom sampling interval
            variance:   filtered observation variance (same and fixed for all params., covariance assumed diagonal)

        Returns:
            instance of OUProcessState
        """
        # TODO: random log-uniform sampling for mu, theta, sigma i.f.f. log_prices are used as in BivariatePriceModel
        sample = dict()
        for name, param, low_threshold in zip(
                ['mu', 'theta', 'sigma', 'driver_df'],
                [mu, theta, sigma, driver_df],
                [-np.inf, 1e-8, 1e-8, 2.999],
        ):
            interval = np.asarray(param)
            assert interval.ndim == 1 and interval[0] <= interval[-1], \
                ' Expected param `{}` as iterable of ordered values as: [lower_bound, upper_bound], got: {}'.format(
                    name, interval
                )
            assert interval[0] > low_threshold, \
                'Expected param `{}` lower bound be bigger than {}, got: {}'.format(name, low_threshold, interval[0])
            sample[name] = np.random.uniform(low=interval[0], high=interval[-1])

        observation = OUEstimatorState(
            mu=sample['mu'],
            log_theta=np.log(sample['theta']),
            log_sigma=np.log(sample['sigma'])
        )
        filtered = CovarianceState(
            mean=np.asarray(observation),
            variance=np.ones(3) * variance,
            covariance=np.eye(3) * variance,
        )
        return OUProcessState(
            observation=observation,
            filtered=filtered,
            driver_df=sample['driver_df'],
        )

    def fit_driver(self, trajectory=None):
        """
        Updates Student-t driver shape parameter. Needs entire trajectory for correct estimation.
        TODO: make recursive update.

        Args:
            trajectory: full observed data of size ~[max_length] or None

        Returns:
            Estimated shape parameter.
        """
        self.ready()
        driver_df, _, _ = self.driver_estimator.fit(trajectory)

        return driver_df

    def reset(self, init_trajectory):
        """
        Resets model parameters for process dX = -Theta *(X - Mu) + Sigma * dW
        and starts new trajectory given initial data.

        Args:
            init_trajectory:    initial 1D process observations trajectory of size [num_points]
        """
        _ = self.stat.reset(init_trajectory[None, :])

        init_observation = np.asarray(self.estimator.reset(init_trajectory))
        # 2x observation to get initial covariance matrix estimate:
        init_observation = np.stack([init_observation, init_observation], axis=-1)
        _ = self.filter.reset(init_observation)

        self.driver_estimator.reset(self.estimator.residuals)
        self.is_ready = True

    def update(self, trajectory, disjoint=False):
        """
        Updates model parameters estimates for process dX = -Theta *(X - Mu) + Sigma * dW
        given new observations.

        Args:
            trajectory:  1D process observations trajectory update of size [num_points]
            disjoint:    bool, indicates whether update given is continuous or disjoint w.r.t. previous one
        """
        self.ready()
        _ = self.stat.update(trajectory[None, :])  # todo: disjoint is ignored or reset stat?

        # Get new state-space observation:
        observation = self.estimator.update(trajectory, disjoint)

        # Smooth and make it random variable:
        _ = self.filter.update(np.asarray(observation)[:, None])

        # Residuals distr. shape update but do not fit:
        self.driver_estimator.update(self.estimator.residuals)

    @staticmethod
    def sample_from_filtered(filter_state, size=1):
        """
        Samples process parameters values given smoothed observations.
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

    @staticmethod
    def sample_naive_unbiased(state, size=1):
        """
        Samples process parameters values given observed values and smoothed covariance.
        Static method, can be used as stand-along function.

        Args:
            state:  instance of OUProcessState
            size:   int or None, number of samples to draw

        Returns:
            sampled process parameters of size [size] each, packed as OUEstimatorState tuple

        """
        assert isinstance(state, OUProcessState), \
            'Expected filter_state as instance of `OUProcessState`, got: {}'.format(type(state))

        # naive_mean = (np.asarray(state.observation) + state.filtered.mean) / 2
        naive_mean = np.asarray(state.observation)
        sample = np.random.multivariate_normal(naive_mean, state.filtered.covariance, size=size)

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

        # return self.sample_from_filtered(state.filtered, size=size)
        return self.sample_naive_unbiased(state, size=size)

    @staticmethod
    def generate_trajectory_fn(batch_size, size, parameters, t_df):
        """
        Generates batch of univariate process realisations given process parameters.
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
        return trajectory.T

    @staticmethod
    def generate_multivariate_trajectory_fn(batch_size, size, parameters, t_df, covariance):
        """
        Generates batch of realisations of multivariate Ornshtein-Uhlenbeck process.
        Note differences in parameters dimensionality w.r.t. univarite case!
        Static method, can be used as stand-along function.

        Args:
            batch_size:     uint, number of trajectories to generates
            size:           uint, trajectory length to generate
            parameters:     instance of OUEstimatorState of size [process_dim] for each parameter
            t_df:           array_like, driver shape param. vector of size [process_dim]
            covariance:     process innovations covariance matrix of size [process_dim, process_dim]

        Returns:
            process realisations as array of size [batch_size, size, process_dim]
        """
        assert isinstance(parameters, OUEstimatorState), \
            'Expected `parameters` as instance of OUEstimatorState, got: {}'.format(type(parameters))

        trajectory = multivariate_ou_process_t_driver_batch_fn(
            batch_size=batch_size,
            num_points=size,
            mu=parameters.mu,
            theta=np.exp(parameters.log_theta),
            sigma=np.exp(parameters.log_sigma),
            cov=covariance,
            df=t_df,
            x0=parameters.mu,
        )
        return trajectory

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
            process realisations of size [batch_size, size]

        """
        self.ready()
        parameters = self.sample_parameters(state, size=batch_size)

        if driver_df is None:
            driver_df, _, _ = self.driver_estimator.fit()
        print('driver_df: ', driver_df)
        return self.generate_trajectory_fn(batch_size, size, parameters, driver_df)


TimeSeriesModelState = namedtuple('TimeSeriesModelState', ['process', 'analyzer'])


class TimeSeriesModel:
    """
    Base time-series modeling and decomposition wrapper class.

    Consist of two [independent] functional parts:
    - stochastic process modeling (fitting and tracking of unobserved parameters, new data generation);
    - realisation trajectory analysis and decomposition;

    Note that there this base class does not include data pre-processing, methods (normalisation, log_transform etc.).
    Later should be added at user_level child classes.

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

    @staticmethod
    def get_random_state(**kwargs):
        """
        Random state sample wrapper.

        Args:
            kwargs:   dict, stochastic process parameters, see kwargs at: OUProcess.get_random_state

        Returns:
            instance of TimeSeriesModelState with `analyser` set to None
        """
        return TimeSeriesModelState(
            process=OUProcess.get_random_state(**kwargs),
            analyzer=None,
        )

    def ready(self):
        assert self.process.is_ready and self.analyzer.is_ready,\
            'TimeSeriesModel is not initialized. Hint: forgot to call .reset()?'

    def reset(self, init_trajectory):
        """
        Resets model parameters and trajectory given initial data.

        Args:
            init_trajectory:    initial time-series observations of size from [1] to [num_points]
        """
        self.process.reset(init_trajectory)
        _ = self.analyzer.reset(init_trajectory)

    def update(self, trajectory, disjoint=False):
        """
        Updates model parameters and trajectory given new data.

        Args:
            trajectory: time-series update observations of size from [1] to [num_points],
                        where num_points <= max_length to keep model trajectory continuous
            disjoint:   bool, indicates whether update given is continuous or disjoint w.r.t. previous one
        """
        _ = self.analyzer.update(trajectory, disjoint)
        self.process.update(trajectory, disjoint)

    def transform(self, trajectory=None, state=None, size=None):
        """
        Returns analyzer data decomposition.

        Args:
            trajectory:     data to decompose of size [num_points] or None
            state:          instance of TimeSeriesModelState or None
            size:           uint, size of decomposition to get, or None

        Returns:
            SSA decomposition of given trajectory w.r.t. given state
            if no `trajectory` is given - returns stored data decomposition
            if no `state` arg. is given - uses stored analyzer state.
            if no 'size` arg is given - decomposes full [stored or given] trajectory
        """
        # Ff 1d signal is given - need to embed it first:
        if trajectory is not None:
            trajectory = np.squeeze(trajectory)
            assert trajectory.ndim == 1, 'Expected 1D array but got shape: {}'.format(trajectory.shape)
            x_embedded = self.analyzer._delay_embed(trajectory, self.analyzer.window)
        else:
            x_embedded = None

        if state is not None:
            assert isinstance(state, TimeSeriesModelState), \
                'Expected `state` as instance of TimeSeriesModelState, got: {}'.format(type(state))
            # Unpack:
            state = state.analyzer

        return self.analyzer.transform(x_embedded, state, size)

    def get_trajectory(self, size=None):
        """
        Returns stored fragment of original time-series data.

        Args:
            size:   uint, fragment length in [1, ..., max_length] or None

        Returns:
            1d series as [ x[-size], x[-size+1], ... x[-1] ], up to length [size];
            if no `size` arg. is given - returns entire stored trajectory, up to length [max_length].
        """
        return self.analyzer.get_trajectory(size)

    def generate(self, batch_size, size, state=None, driver_df=None):
        """
        Generates batch of realisations given process parameters.

        Args:
            batch_size:     uint, number of realisations to draw
            size:           uint, length of each one
            state:          instance TimeSeriesModelState or None, model parameters to use
            driver_df:      t-student process driver degree of freedom parameter or None

        Returns:
            process realisations batch of size [batch_size, size]
        """
        if state is not None:
            assert isinstance(state, TimeSeriesModelState), \
                'Expected `state` as instance of TimeSeriesModelState, got: {}'.format(type(state))
            # Unpack:
            process_state = state.process
            if driver_df is None:
                # Get driver param from given state:
                driver_df = state.process.driver_estimator.df
        else:
            process_state = None

        return self.process.generate(batch_size, size, process_state, driver_df)


PriceModelState = namedtuple('PriceModelState', ['process', 'analyzer', 'stat'])


class PriceModel(TimeSeriesModel):
    """
    Wrapper class for positive-valued time-series.
    Internally works with normalised log-transformed data.
    """
    def __init__(
            self,
            max_length,
            analyzer_window,
            analyzer_grouping=None,
            alpha=None,
            filter_alpha=None,
            stat_alpha=None
    ):
        """

        Args:
            max_length:         uint, maximum trajectory length to keep;
            analyzer_window:    uint, SSA embedding window;
            analyzer_grouping:  SSA decomposition triples grouping,
                                iterable of pairs convertible to python slices, i.e.:
                                grouping=[[0,1], [1,2], [2, None]];
            alpha:              float in [0, 1], SSA and process estimator decaying factor;
            filter_alpha:       float in [0, 1], process smoothing decaying factor;
            stat_alpha:         float in [0, 1], time-series statistics tracking decaying factor;
        """
        super().__init__(max_length, analyzer_window, analyzer_grouping, alpha, filter_alpha)

        # Statistics of original data:
        self.stat = Zscore(1, stat_alpha)

    def get_state(self):
        """
        Returns model state tuple.

        Returns:
            current state as instance of PriceModelState
        """
        return PriceModelState(
            process=self.process.get_state(),
            analyzer=self.analyzer.get_state(),
            stat=self.stat.get_state(),
        )

    @staticmethod
    def normalise(trajectory, mean, variance):
        return (trajectory - mean) / np.clip(variance, 1e-8, None) ** .5

    @staticmethod
    def denormalize(trajectory, mean, variance):
        return trajectory * variance ** .5 + mean

    def reset(self, init_trajectory):
        """
        Resets model parameters and trajectory given initial data.

        Args:
            init_trajectory:    initial time-series observations of size from [1] to [num_points]
        """
        log_data = np.log(init_trajectory)
        mean, variance = self.stat.reset(log_data[None, :])
        return super().reset(self.normalise(log_data, mean, variance))

    def update(self, trajectory, disjoint=False):
        """
        Updates model parameters and trajectory given new data.

        Args:
            trajectory: time-series update observations of size from [1] to [num_points],
                        where num_points <= max_length to keep model trajectory continuous
            disjoint:   bool, indicates whether update given is continuous or disjoint w.r.t. previous one
        """
        log_data = np.log(trajectory)
        mean, variance = self.stat.update(log_data[None, :])
        return super().update(self.normalise(log_data, mean, variance), disjoint)

    def transform(self, trajectory=None, state=None, size=None):
        """
        Returns analyzer data decomposition.

        Args:
            trajectory:     data to decompose of size [num_points] or None
            state:          instance of PriceModelState or None
            size:           uint, size of decomposition to get, or None

        Returns:
            SSA decomposition of given trajectory w.r.t. given state
            if no `trajectory` is given - returns stored data decomposition
            if no `state` arg. is given - uses stored analyzer state.
            if no 'size` arg is given - decomposes full [stored or given] trajectory
        """
        if state is not None:
            assert isinstance(state, PriceModelState), \
                'Expected `state` as instance of PriceModelState, got: {}'.format(type(state))
            # Unpack state:
            state_base = TimeSeriesModelState(
                analyzer=state.analyzer,
                process=state.process
            )
        else:
            state_base = None

        # If 1d signal is given - need to normalize:
        if trajectory is not None:
            assert state is not None, 'State is expected when trajectory is given'
            trajectory = self.normalise(np.log(trajectory), state.stat.mean, state.stat.variance)

        return super().transform(trajectory, state_base, size)

    def get_trajectory(self, size=None):
        """
        Returns stored fragment of original time-series data.

        Args:
            size:   uint, fragment length in [1, ..., max_length] or None

        Returns:
            1d series as [ x[-size], x[-size+1], ... x[-1] ], up to length [size];
            if no `size` arg. is given - returns entire stored trajectory, up to length [max_length].
        """
        # TODO: reconstruction is freaky due to only last stored statistic is used
        trajectory = super().get_trajectory(size)
        state = self.get_state()

        return np.exp(self.denormalize(trajectory, state.stat.mean, state.stat.variance))

    def generate(self, batch_size, size, state=None, driver_df=None):
        """
        Generates batch of realisations given process parameters.

        Args:
            batch_size:     uint, number of realisations to draw
            size:           uint, length of each one
            state:          instance PriceModelState or None, model parameters to use
            driver_df:      t-student process driver degree of freedom parameter or None

        Returns:
            process realisations batch of size [batch_size, size]
        """
        if state is not None:
            assert isinstance(state, PriceModelState), \
                'Expected `state` as instance of PriceModelState, got: {}'.format(type(state))
            # Unpack:
            state_base = TimeSeriesModelState(
                analyzer=state.analyzer,
                process=state.process
            )
        else:
            state = self.get_state()
            state_base = None

        trajectory = super().generate(batch_size, size, state_base, driver_df)

        return np.exp(self.denormalize(trajectory, state.stat.mean, state.stat.variance))

    @staticmethod
    def get_random_state(p_params, mean=(100, 100), variance=(1, 1)):
        """
        Samples random uniform model state w.r.t. intervals given.

        Args:
            p_params:       dict, stochastic process parameters, see kwargs at: OUProcess.get_random_state
            mean:           iterable of floats as [0 < lower_bound, upper_bound], time-series means sampling interval.
            variance:       iterable of floats as [0 < lower_bound, upper_bound], time-series variances sampling interval.

        Returns:
            instance of PriceModelState with `analyser` set to None

        Note:
            negative means are rejected;
            stochastic process fitted on log_normalized data;
        """
        sample = dict()
        for name, param, low_threshold in zip(
                ['mean', 'variance', ], [mean, variance], [1e-8, 1e-8]):
            interval = np.asarray(param)
            assert interval.ndim == 1 and interval[0] <= interval[-1], \
                ' Expected param `{}` as iterable of ordered values as: [lower_bound, upper_bound], got: {}'.format(
                    name, interval
                )
            assert interval[0] >= low_threshold, \
                'Expected param `{}` lower bound be no less than {}, got: {}'.format(name, low_threshold, interval[0])

            sample[name] = np.random.uniform(low=interval[0], high=interval[-1], size=1)

        # Log_transform mean and variance (those is biased estimates but ok for rnd. samples):
        log_variance = np.log(sample['variance'] / sample['mean'] ** 2 + 1)
        log_mean = np.log(sample['mean']) - .5 * log_variance

        # Inverse transform memo:
        # mean = exp(log_mean + 0.5 * log_var)
        # var = mean**2 * (exp(log_var) -1)

        return PriceModelState(
            process=OUProcess.get_random_state(**p_params),
            analyzer=None,
            stat=ZscoreState(
                mean=log_mean,
                variance=log_variance
            ),
        )

