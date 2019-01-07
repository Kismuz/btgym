import numpy as np
import copy
import pandas as pd

from btgym.datafeed.derivative import BTgymDataset2
from btgym.datafeed.multi import BTgymMultiData

from btgym.research.model_based.datafeed.base import BasePairDataGenerator, BaseCombinedDataSet
from btgym.research.model_based.model.deprecated.pfd import PairFilteredModel, PairModelState


def simple_filtered_decomp_generator_fn(num_points, state, keep_decimals=6, **kwargs):
    assert isinstance(state, PairModelState), \
        'Expected model state as instance of `PairModelState`, got: {}'.format(type(state))
    _, x = PairFilteredModel.generate_trajectory_fn(1, num_points, state, u=PairFilteredModel.u_recon)
    return np.around(x, decimals=keep_decimals)


def simple_filtered_decomp_parameters_fn(state_trajectory):
    idx = np.random.randint(0, len(state_trajectory))
    z = state_trajectory[idx]
    return dict(
        state=z,
        # for tf.summaries via strategy:
        ou_mu=z.s_mean[0],
        ou_lambda=np.exp(z.s_mean[1]),
        ou_sigma=np.exp(z.s_mean[2]),
        x0=0.0,
    )


class SimplePFDGenerator(BasePairDataGenerator):
    """
    Generates O=H=L=C data driven by Filtered Decomposition Model
    """
    def __init__(
            self,
            data_names,
            generator_parameters_config,
            generator_fn=simple_filtered_decomp_generator_fn,
            generator_parameters_fn=simple_filtered_decomp_parameters_fn,
            name='sPFDGenerator',
            **kwargs

    ):
        super(SimplePFDGenerator, self).__init__(
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
        }

    def generate_data(self, generator_params, sample_type=0):
        """
        Generates data trajectory.

        Args:
            generator_params:       dict, data_generating_function parameters
            sample_type:            0 - generate train data | 1 - generate test data

        Returns:
            data as two pandas dataframes
        """
        # Get data shaped [1, 2, num_points] and map to OHLC pattern:
        data = self.generator_fn(num_points=self.data[self.a1_name].episode_num_records, **generator_params)
        p1_dict = {
            'mean': data[0, 0, :],
            'maximum': data[0, 0, :],
            'minimum': data[0, 0, :],
            'last': data[0, 0, :],
        }
        p2_dict = {
            'mean': data[0, 1, :],
            'maximum': data[0, 1, :],
            'minimum': data[0, 1, :],
            'last': data[0, 1, :],
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
        Overrides base method by employing single underlying stochastic process to generate two tragectories
        Args:
            sample_type:    bool, train/test
            **kwargs:

        Returns:
            sample as SimplePFDGenerator instance
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
        sample = SimplePFDGenerator(
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


class SimplePFDataSet(BaseCombinedDataSet):
    """
    Combined data iterator provides:
    Train:
        two trajectories of OHLC prices modelled by 'Pair Filtered Decomposition' model;
        High-Low spread values not modelled -> Simple name;

    Test:
        two real historic time-consistent data lines;

    """
    def __init__(
            self,
            assets_filenames,
            state_trajectory,
            train_episode_duration=None,
            test_episode_duration=None,
            name='SimplePFData',
            **kwargs
    ):
        """

        Args:
            assets_filenames:           dict of two keys in form of {'asset_name`: 'data_file_name'}, test data
            state_trajectory:           iterable of PairModelState instances
            train_episode_duration:     dict of keys {'days', 'hours', 'minutes'} - train sample duration
            test_episode_duration:      dict of keys {'days', 'hours', 'minutes'} - test sample duration
        """
        assert isinstance(assets_filenames, dict), \
            'Expected `assets_filenames` type `dict`, got {} '.format(type(assets_filenames))

        data_names = [name for name in assets_filenames.keys()]
        assert len(data_names) == 2, 'Expected exactly two assets, got: {}'.format(data_names)

        assert isinstance(assets_filenames, dict), \
            'Expected `assets_filenames` type `dict`, got {} '.format(type(assets_filenames))

        data_names = [name for name in assets_filenames.keys()]
        assert len(data_names) == 2, 'Expected exactly two assets, got: {}'.format(data_names)

        generator_parameters_config = dict(state_trajectory=state_trajectory)

        train_data_config = dict(
            data_names=data_names,
            generator_parameters_config=generator_parameters_config,
            episode_duration=train_episode_duration,
        )
        test_data_config = dict(
            data_class_ref=BTgymDataset2,
            data_config={asset_name: {'filename': file_name} for asset_name, file_name in assets_filenames.items()},
            episode_duration=test_episode_duration,
        )
        super(SimplePFDataSet, self).__init__(
            train_data_config=train_data_config,
            test_data_config=test_data_config,
            train_class_ref=SimplePFDGenerator,
            test_class_ref=BTgymMultiData,
            name=name,
            **kwargs
        )
