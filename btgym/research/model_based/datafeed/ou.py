###############################################################################
#
# Copyright (C) 2017, 2018 Andrew Muzikin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

import copy
import pandas as pd

from .base import BaseCombinedDataSet, BasePairCombinedDataSet, BasePairDataGenerator
from .base import base_random_uniform_parameters_fn, base_spread_generator_fn
from btgym.research.model_based.model.stochastic import ornshtein_uhlenbeck_process_fn, ornshtein_uhlenbeck_uniform_parameters_fn
from btgym.research.model_based.model.stochastic import ornshtein_uhlenbeck_log_uniform_parameters_fn
from btgym.research.model_based.model.stochastic import weiner_process_fn, weiner_process_uniform_parameters_fn
from btgym.research.model_based.model.stochastic import coupled_wave_pair_generator_fn
from btgym.datafeed.derivative import BTgymDataset2
from btgym.datafeed.multi import BTgymMultiData


class UniformOUGenerator(BaseCombinedDataSet):
    """
    Combined data iterator provides:
    - realisations of Ornstein-Uhlenbeck process as train data;
    - real historic data as test data;

    OUp. paramters are randomly uniformly sampled from given intervals.
    """
    def __init__(self, ou_mu, ou_lambda, ou_sigma, ou_x0=None, name='UniformOUData', **kwargs):
        """

        Args:
            ou_mu:                      float or iterable of 2 floats, Ornstein-Uhlenbeck process mean value or interval
            ou_lambda:                  float or iterable of 2 floats, OUp. mean-reverting rate or interval
            ou_sigma:                   float or iterable of 2 floats, OUp. volatility value or interval
            ou_x0:                      float or iterable of 2 floats, OUp. trajectory start value or interval
            filename:                   str, test data filename
            parsing_params:             dict test data parsing params
            episode_duration_train:     dict, duration of train episode in days/hours/mins
            episode_duration_test:      dict, duration of test episode in days/hours/mins
            time_gap:                   dict test episode duration tolerance
            start_00:                   bool, def=False
            timeframe:                  int, data periodicity in minutes
            name:                       str
            data_names:                 iterable of str
            global_time:                dict {y, m, d} to set custom global time (here for plotting only)
            task:                       int
            log_level:                  logbook.Logger level
        """
        super(UniformOUGenerator, self).__init__(
            generator_fn=ornshtein_uhlenbeck_process_fn,
            generator_parameters_fn=ornshtein_uhlenbeck_uniform_parameters_fn,
            generator_parameters_config={'mu': ou_mu, 'l': ou_lambda, 'sigma': ou_sigma, 'x0': ou_x0},
            name=name,
            **kwargs
        )


class LogUniformOUGenerator(BaseCombinedDataSet):
    """
    Combined data iterator provides:
    - realisations of Ornstein-Uhlenbeck process as train data;
    - real historic data as test data;

    Lambda parameter is sampled from log-uniform distribution,
    Sigma, Mu parameters are sampled from uniform distributions defined by given intervals.
    """
    def __init__(self, ou_mu, ou_lambda, ou_sigma, ou_x0=None, name='LogUniformOUData', **kwargs):
        """

        Args:
            ou_mu:                      float or iterable of 2 floats, Ornstein-Uhlenbeck process mean value or interval
            ou_lambda:                  float or iterable of 2 floats, OUp. mean-reverting rate or interval
            ou_sigma:                   float or iterable of 2 floats, OUp. volatility value or interval
            ou_x0:                      float or iterable of 2 floats, OUp. trajectory start value or interval
            filename:                   str, test data filename
            parsing_params:             dict test data parsing params
            episode_duration_train:     dict, duration of train episode in days/hours/mins
            episode_duration_test:      dict, duration of test episode in days/hours/mins
            time_gap:                   dict test episode duration tolerance
            start_00:                   bool, def=False
            timeframe:                  int, data periodicity in minutes
            name:                       str
            data_names:                 iterable of str
            global_time:                dict {y, m, d} to set custom global time (here for plotting only)
            task:                       int
            log_level:                  logbook.Logger level
        """
        super(LogUniformOUGenerator, self).__init__(
            generator_fn=ornshtein_uhlenbeck_process_fn,
            generator_parameters_fn=ornshtein_uhlenbeck_log_uniform_parameters_fn,
            generator_parameters_config={'mu': ou_mu, 'l': ou_lambda, 'sigma': ou_sigma, 'x0': ou_x0},
            name=name,
            **kwargs
        )


class OUGenerator(BaseCombinedDataSet):
    """
    Combined data iterator provides:
    - realisations of Ornstein-Uhlenbeck process as train data;
    - real historic data as test data;

    This class expects OUp. parameters no-args callable to be explicitly provided.
    """
    def __init__(self, generator_parameters_fn, name='OUData', **kwargs):
        """

        Args:
            generator_parameters_fn:    callable, should return dictionary of generator_fn kwargs: {l, mu, sigma};
                                        this callable itself should not require any args
            filename:                   str, test data filename
            parsing_params:             dict test data parsing params
            episode_duration_train:     dict, duration of train episode in days/hours/mins
            episode_duration_test:      dict, duration of test episode in days/hours/mins
            time_gap:                   dict test episode duration tolerance
            start_00:                   bool, def=False
            timeframe:                  int, data periodicity in minutes
            name:                       str
            data_names:                 iterable of str
            global_time:                dict {y, m, d} to set custom global time (here for plotting only)
            task:                       int
            log_level:                  logbook.Logger level
        """
        super(OUGenerator, self).__init__(
            generator_fn=ornshtein_uhlenbeck_process_fn,
            generator_parameters_fn=generator_parameters_fn,
            name=name,
            **kwargs
        )


class PairOUDataSet(BasePairCombinedDataSet):
    """
    Combined data iterator provides:
    Train:
        two integrated synthetic data lines composed as:
        line2 = Weiner_tragectory + .5 * OU_tragectory
        line2 = Weiner_tragectory - .5 * OU_tragectory

    Test:
        two real historic time-consistent data lines;
    """

    def __init__(
            self,
            assets_filenames,
            ou_lambda,
            ou_sigma,
            ou_mu,
            weiner_delta,
            x0,
            spread_alpha,
            spread_beta,
            spread_max,
            spread_min,
            name='PairedOuDataSet',
            **kwargs
    ):
        """

        Args:
            assets_filenames:           (req.) dict of str, test data filenames (two files expected)
            ou_mu:                      float or iterable of 2 floats, Ornstein-Uhlenbeck process mean value or interval
            ou_lambda:                  float or iterable of 2 floats, OUp. mean-reverting rate or interval
            ou_sigma:                   float or iterable of 2 floats, OUp. volatility value or interval
            ou_x0:                      float or iterable of 2 floats, OUp. trajectory start value or interval
            weiner_delta:               float or iterable of 2 floats, Weiner p. trajectory speed parameter or interval
            spread_alpha:               float, random spread beta-distributed alpha param
            spread_beta:                float, random spread beta-distributed beta param
            spread_max:                 float, random spread min. value
            spread_min:                 float, random spread max. value
            train_episode_duration:     dict, duration of train episode in days/hours/mins
            test_episode_duration:      dict, duration of test episode in days/hours/mins
        """
        process_1_config = dict(
            generator_fn=weiner_process_fn,
            generator_parameters_fn=weiner_process_uniform_parameters_fn,
            generator_parameters_config={'delta': weiner_delta, 'x0': x0},
            spread_generator_fn=base_spread_generator_fn,
            spread_generator_parameters={
                'alpha': spread_alpha,
                'beta': spread_beta,
                'minimum': spread_min,
                'maximum': spread_max
            },
        )
        process_2_config = dict(
            generator_fn=ornshtein_uhlenbeck_process_fn,
            generator_parameters_fn=ornshtein_uhlenbeck_uniform_parameters_fn,
            generator_parameters_config={'mu': ou_mu, 'l': ou_lambda, 'sigma': ou_sigma, 'x0': None},
            spread_generator_fn=None,
        )
        super(PairOUDataSet, self).__init__(
            assets_filenames=assets_filenames,
            process1_config=process_1_config,
            process2_config=process_2_config,
            name=name,
            **kwargs
        )


class PairWaveModelGenerator(BasePairDataGenerator):
    """
    More-or-less realistic OHLC model.
    Utilizes single stochastic model to generate two integrated trajectories.
    """
    def __init__(
            self,
            data_names,
            generator_parameters_config,
            generator_fn=coupled_wave_pair_generator_fn,
            generator_parameters_fn=base_random_uniform_parameters_fn,
            name='PairWaveModelGenerator',
            **kwargs

    ):
        super(PairWaveModelGenerator, self).__init__(
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
        # Get data shaped [2, 4, num_points] and map to OHLC pattern:
        data = self.generator_fn(num_points=self.data[self.a1_name].episode_num_records, **generator_params)
        p1_dict = {
            'mean': data[0, 0, :],
            'maximum': data[0, 1, :],
            'minimum': data[0, 2, :],
            'last': data[0, 3, :],
        }
        p2_dict = {
            'mean': data[1, 0, :],
            'maximum': data[1, 1, :],
            'minimum': data[1, 2, :],
            'last': data[1, 3, :],
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
            sample as PairWaveModelGenerator instance
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
        sample = PairWaveModelGenerator(
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


class PairWaveModelDataSet(BaseCombinedDataSet):
    """
    Combined data iterator provides:
    Train:
        two trajectories of OHLC prices modelled by OU process with stochastic drift;
        High-Low spread values for each price line independently generated by 'coupled wave model';

    Test:
        two real historic time-consistent data lines;

    """
    def __init__(
            self,
            assets_filenames,
            drift_sigma,
            ou_sigma,
            ou_lambda,
            ou_mu,
            spread_sigma_1,
            spread_sigma_2,
            spread_mean_1,
            spread_mean_2,
            bias,
            train_episode_duration=None,
            test_episode_duration=None,
            name='PairedWaveData',
            **kwargs
    ):
        """

        Args:
            assets_filenames:           dict of two keys in form of {'asset_name`: 'data_file_name'}, test data
            drift_sigma:                ufloat, stichastic drift sigma
            ou_sigma:                   ufloat, base OU process sigma
            ou_lambda:                  ufloat, base OU mean-reverting speed parameter
            ou_mu:                      float, base OU mean parameter
            spread_sigma_1:             ufloat, Hi-Lo spread generating sigma1
            spread_sigma_2:             ufloat, Hi-Lo spread generating sigma2
            spread_mean_1:              float, Hi-Lo spread generating mean1
            spread_mean_2:              float, Hi-Lo spread generating mean2
            bias:                       ufloat, process starting point
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

        generator_parameters_config = dict(
            drift_sigma=drift_sigma,
            ou_sigma=ou_sigma,
            ou_lambda=ou_lambda,
            ou_mu=ou_mu,
            spread_sigma_1=spread_sigma_1,
            spread_sigma_2=spread_sigma_2,
            spread_mean_1=spread_mean_1,
            spread_mean_2=spread_mean_2,
            bias=bias,
        )
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
        super(PairWaveModelDataSet, self).__init__(
            train_data_config=train_data_config,
            test_data_config=test_data_config,
            train_class_ref=PairWaveModelGenerator,
            test_class_ref=BTgymMultiData,
            name=name,
            **kwargs
        )


