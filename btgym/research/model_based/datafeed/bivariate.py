import numpy as np
import copy
import pandas as pd
import random

from btgym.datafeed.derivative import BTgymDataset2
from btgym.datafeed.multi import BTgymMultiData

from btgym.research.model_based.datafeed.base import BasePairDataGenerator, BaseCombinedDataSet
from btgym.research.model_based.model.bivariate import BivariatePriceModel


def bivariate_generator_fn(num_points, state, keep_decimals=6, **kwargs):
    """
    Wrapper around data modeling class generative method.

    Args:
        num_points:         trajectory length to draw
        state:              model state, instance of BivariateTSModelState
        keep_decimals:      number of decimal places to round to

    Returns:
        generated time-series of size [1, 2, size]
    """
    _, x = BivariatePriceModel.generate_bivariate_trajectory_fn(1, num_points, state, True, BivariatePriceModel.u_recon)
    return np.around(x, decimals=keep_decimals)


def bivariate_random_state_fn(*args, **kwargs):
    """
    Samples random model state.
    Args:
        *args:          same as args for: BivariateTSModel.get_random_state
        **kwargs:       same as args for: BivariateTSModel.get_random_state

    Returns:
        dictionary holding instance of BivariateTSModelState and auxillary fields
    """
    state = BivariatePriceModel.get_random_state(*args, **kwargs)
    return dict(
        state=state,
        # for tf.summaries via strategy:
        ou_mu=state.s.process.observation.mu,
        ou_lambda=np.exp(state.s.process.observation.log_theta),
        ou_sigma=np.exp(state.s.process.observation.log_sigma),
        x0=state.s.process.observation.mu,
    )


def bivariate_state_set_iterator_fn(states_set, **kwargs):
    """
    Randomly sample state from given set of states.

    Args:
        states_set:     iterable containing instances of BivariateTSModelState

    Returns:
        dictionary holding instance of BivariateTSModelState and auxillary fields
    """
    state = random.choice(tuple(states_set))

    return dict(
        state=state,
        # for tf.summaries via strategy:
        ou_mu=state.s.process.observation.mu,
        ou_lambda=np.exp(state.s.process.observation.log_theta),
        ou_sigma=np.exp(state.s.process.observation.log_sigma),
        x0=state.s.process.observation.mu,
    )


class BivariateGenerator(BasePairDataGenerator):
    """
    Generates O=H=L=C price data driven by `BivariatePriceModel`
    """
    def __init__(
            self,
            data_names,
            generator_parameters_config,
            generator_fn=bivariate_generator_fn,
            generator_parameters_fn=bivariate_random_state_fn,
            name='BivariateGenerator',
            **kwargs

    ):
        super(BivariateGenerator, self).__init__(
            data_names,
            process1_config=None,  # bias generator
            process2_config=None,  # spread generator
            name=name,
            **kwargs
        )
        self.generator_fn = generator_fn
        self.generator_parameters_fn = generator_parameters_fn
        self.generator_parameters_config = generator_parameters_config

        self.columns_map = {
            'open': 'mean',
            'high': 'maximum',
            'low': 'minimum',
            'close': 'last',
            'bid': 'minimum',
            'ask': 'maximum',
            'mid': 'mean',
            'volume': 'nothing',
        }

    def generate_data(self, generator_params, sample_type=0):
        """
        Generates data trajectory.

        Args:
            generator_params:       dict, data_generating_function parameters
            sample_type:            0 - generate train data | 1 - generate test data

        Returns:
            tuple of two pandas.dataframe objects
        """
        # Get data shaped [1, 2, num_points] and map to OHLC pattern:
        data = self.generator_fn(num_points=self.data[self.a1_name].episode_num_records, **generator_params)

        # No fancy OHLC modelling yet:
        p1_dict = {
            'mean': data[0, 0, :],
            'maximum': data[0, 0, :],
            'minimum': data[0, 0, :],
            'last': data[0, 0, :],
            'nothing': np.zeros(data[0, 1, :].shape)
        }
        p2_dict = {
            'mean': data[0, 1, :],
            'maximum': data[0, 1, :],
            'minimum': data[0, 1, :],
            'last': data[0, 1, :],
            'nothing': np.zeros(data[0, 1, :].shape)
        }
        # Make dataframes:
        if sample_type:
            index = self.data[self.a1_name].test_index
        else:
            index = self.data[self.a1_name].train_index
        # Map dictionary of data to dataframe columns:
        df1 = pd.DataFrame(data={name: p1_dict[self.columns_map[name]] for name in self.names}, index=index)
        df2 = pd.DataFrame(data={name: p2_dict[self.columns_map[name]] for name in self.names}, index=index)

        return df1, df2

    def sample(self, sample_type=0, broadcast_message=None, **kwargs):
        """
        Generates pair of price trajectories and pack it as DataSet_type object.

        Args:
            sample_type:        bool, train/test
            broadcast_message:  <reserved for future param.>
            **kwargs:

        Returns:
            sample as SimpleBivariateGenerator instance
        """
        # self.log.debug('broadcast_message: <<{}>>'.format(broadcast_message))

        if self.metadata['type'] is not None:
            if self.metadata['type'] != sample_type:
                self.log.warning(
                    'Attempt to sample type {} given current sample type {}, overriden.'.format(
                        sample_type,
                        self.metadata['type']
                    )
                )
                sample_type = self.metadata['type']

        # Prepare empty instance of multi_stream data:
        sample = type(self)(
            data_names=self.data_names,
            generator_parameters_config=self.generator_parameters_config,
            data_class_ref=self.data_class_ref,
            name='sub_' + self.name,
            _top_level=False,
            **self.nested_kwargs
        )
        # TODO: WTF?
        sample.names = self.names

        if self.get_new_sample:
            # get parameters:
            params = self.generator_parameters_fn(**self.generator_parameters_config)

            data1, data2 = self.generate_data(params, sample_type=sample_type)

            metadata = {'generator': params}

        else:
            data1 = None
            data2 = None
            metadata = {}

        metadata.update(
            {
                'type': sample_type,
                'sample_num': self.sample_num,
                'parent_sample_type': self.metadata['type'],
                'parent_sample_num': self.sample_num,
                'first_row': 0,
                'last_row': self.data[self.a1_name].episode_num_records,
            }
        )

        sample.metadata = copy.deepcopy(metadata)

        # Populate sample with data:
        sample.data[self.a1_name].data = data1
        sample.data[self.a2_name].data = data2

        sample.filename = {key: stream.filename for key, stream in self.data.items()}
        self.sample_num += 1
        return sample


class BivariateDataSet(BaseCombinedDataSet):
    """
    Combined data iterator provides:
    - train data as two trajectories of OHLC prices generated by 'BivariatePriceModel';
    - test data as two historic timeindex-matching OHLC data lines;

    """

    def __init__(
            self,
            model_params,
            assets_filenames=None,
            assets_dataframes=None,
            train_episode_duration=None,
            test_episode_duration=None,
            name='BivariateDataSet',
            _train_class_ref=BivariateGenerator,
            _test_class_ref=BTgymMultiData,
            **kwargs
    ):
        """

        Args:
            model_params:               dict holding generative model parameters,
                                        same as kwargs for: BivariatePriceModel.get_random_state() method
            assets_filenames:           dict. of two keys in form of: {'asset_name`: 'data_file_name'},
                                        test data or None, ignored when `assets_dataframes` arg. is given
            assets_dataframes:          dict. of two keys in form of {'asset_name`: pd.dataframe},
                                        an alternative way to provide test data as pandas.dataframes instances,
                                        overrides `assets_filenames`
            train_episode_duration:     dict of keys {'days', 'hours', 'minutes'} - train sample duration
            test_episode_duration:      dict of keys {'days', 'hours', 'minutes'} - test sample duration
        """
        if assets_dataframes is None:
            assert isinstance(assets_filenames, dict), \
                'Expected `assets_filenames` type `dict`, got {} '.format(type(assets_filenames))

            data_config = {
                asset_name: {'filename': f_name, 'dataframe': None} for asset_name, f_name in assets_filenames.items()}

        else:
            assert assets_dataframes is not None, 'Expected either `assets_filenames` or `assets_dataframes`'
            assert isinstance(assets_dataframes, dict), \
                'Expected `assets_filenames` type `dict`, got {} '.format(type(assets_filenames))

            data_config = {
                asset_name: {'dataframe': df, 'filename': None} for asset_name, df in assets_dataframes.items()
            }

        data_names = [name for name in data_config.keys()]
        assert len(data_names) == 2, 'Expected exactly two assets, got: {}'.format(data_names)

        self.train_class_ref = _train_class_ref
        self.test_class_ref = _test_class_ref

        train_data_config = dict(
            data_names=data_names,
            generator_parameters_config=model_params,
            episode_duration=train_episode_duration,
        )
        test_data_config = dict(
            data_class_ref=BTgymDataset2,
            data_config=data_config,
            episode_duration=test_episode_duration,
        )
        super(BivariateDataSet, self).__init__(
            train_data_config=train_data_config,
            test_data_config=test_data_config,
            train_class_ref=self.train_class_ref,
            test_class_ref=self.test_class_ref,
            name=name,
            **kwargs
        )


class BivariateStateSetGenerator(BivariateGenerator):

    def __init__(
            self,
            data_names,
            generator_parameters_config,
            name='BivariateStateSetGenerator',
            **kwargs

    ):
        super().__init__(
            data_names,
            generator_parameters_config,
            generator_fn=bivariate_generator_fn,
            generator_parameters_fn=bivariate_state_set_iterator_fn,
            name=name,
            **kwargs
        )


class BivariateStateDataSet(BivariateDataSet):

    def __init__(self, *args, **kwargs):
        """

        Args:
            model_params:               dict holding generative model parameters,
                                        same as args for: bivariate_state_set_iterator_fn
            assets_filenames:           dict. of two keys in form of: {'asset_name`: 'data_file_name'},
                                        test data or None, ignored when `assets_dataframes` arg. is given
            assets_dataframes:          dict. of two keys in form of {'asset_name`: pd.dataframe},
                                        an alternative way to provide test data as pandas.dataframes instances,
                                        overrides `assets_filenames`
            train_episode_duration:     dict of keys {'days', 'hours', 'minutes'} - train sample duration
            test_episode_duration:      dict of keys {'days', 'hours', 'minutes'} - test sample duration
        """
        super().__init__(*args, _train_class_ref=BivariateStateSetGenerator, **kwargs)

