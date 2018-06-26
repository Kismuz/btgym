from logbook import Logger, StreamHandler, WARNING

import datetime
import random
from numpy.random import beta as random_beta
import copy
import os
import sys

import backtrader.feeds as btfeeds
import pandas as pd


class BTgymMultiData:
    """
    Multiply data streams wrapper.
    """

    def __init__(
            self,
            data_class_ref=None,
            data_config=None,
            name='multi_data',
            task=0,
            log_level=WARNING,
            **kwargs
    ):
        """

        Args:
            data_class_ref:         one of BTgym single-stream datafeed classes
            data_config:        nested dictionary of individual data streams such as:
                                    data_config={
                                        data_name_1: {
                                            filename: [source csv filename string or list of strings],
                                            config: {dict of individual stream config. params}
                                        },
                                        ...,
                                        data_name_n : {...}
                                    }
            kwargs:             shared parameters for all data streams
        """
        self.data_class_ref = data_class_ref
        if data_config is None:
            self.data_config = {}

        else:
            self.data_config = data_config

        self.master_data = None
        self.name = name
        self.task = task
        self.metadata = {'sample_num': 0, 'type': None}
        self.filename = None
        self.is_ready = False
        self.global_timestamp = 0
        self.log_level = log_level
        self.params = {}
        self.names = []

        # Logging:
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('{}_{}'.format(self.name, self.task), level=self.log_level)

        # Make dictionary of single-stream datasets:
        self.data = {}
        for key, stream in self.data_config.items():
            try:
                stream['config'].update(kwargs)

            except KeyError:
                stream['config'] = kwargs

            self.data[key] = self.data_class_ref(
                filename=stream['filename'],
                task=task,
                name='{}_{}'.format(name, key),
                log_level=log_level,
                **stream['config']
            )
            try:
                # If master-data has been pointed explicitly by 'base' kwarg:
                if stream['base']:
                    self.master_data = self.data[key]

            except KeyError:
                pass

    def set_logger(self, level=None, task=None):
        """
        Sets logbook logger.

        Args:
            level:  logbook.level, int
            task:   task id, int

        """
        if task is not None:
            self.task = task

        if level is not None:
            for stream in self.data.values():
                stream.log = Logger('{}_{}'.format(stream.name, stream.task), level=level)

            self.log = Logger('{}_{}'.format(self.name, self.task), level=level)

    def set_params(self, params_dict):
        """
        Batch attribute setter.

        Args:
            params_dict: dictionary of parameters to be set as instance attributes.
        """
        for key, value in params_dict.items():
            for stream in self.data.values():
                setattr(stream, key, value)

    def read_csv(self, data_filename=None, force_reload=False):
        # Load:
        indexes = []
        for stream in self.data.values():
            stream.read_csv(force_reload=force_reload)
            indexes.append(stream.data.index)

        # Get indexes intersection:
        if len(indexes) > 1:
            idx_intersected = indexes[0]
            for i in range(1, len(indexes)):
                idx_intersected = idx_intersected.intersection(indexes[i])

            # Truncate data to common index:
            for stream in self.data.values():
                stream.data = stream.data.loc[idx_intersected]

    def reset(self, **kwargs):
        indexes = []
        for stream in self.data.values():
            stream.reset(**kwargs)
            indexes.append(stream.data.index)

        # Get indexes intersection:
        if len(indexes) > 1:
            idx_intersected = indexes[0]
            for i in range(1, len(indexes)):
                idx_intersected = idx_intersected.intersection(indexes[i])

            idx_intersected.drop_duplicates()
            self.log.info('shared num. records: {}'.format(len(idx_intersected)))

            # Truncate data to common index:
            for stream in self.data.values():
                stream.data = stream.data.loc[idx_intersected]

            # Choose master_data
            if self.master_data is None:
                # Just choose first key:
                all_keys = list(self.data.keys())
                if len(all_keys) > 0:
                    self.master_data = self.data[all_keys[0]]

            self.global_timestamp = self.master_data.global_timestamp
            self.names = self.master_data.names

        self.is_ready = True

    def set_global_timestamp(self, timestamp):
        for stream in self.data.values():
            stream.set_global_timestamp(timestamp)

        self.global_timestamp = self.master_data.global_timestamp

    def describe(self):
        return {key: stream.describe() for key, stream in self.data.items()}

    def sample(self, **kwargs):

        # Get sample to infer exact interval:
        master_sample = self.master_data.sample(**kwargs)

        # Prepare empty instance of multistream data:
        sample = BTgymMultiData(task=self.task, log_level=self.log_level, name='sub_' + self.name)
        sample.metadata = copy.deepcopy(master_sample.metadata)

        kwargs['interval'] = [sample.metadata['first_row'], sample.metadata['last_row']]

        # Populate sample with data:
        for key, stream in self.data.items():
            sample.data[key] = stream.sample(force_interval=True, **kwargs)

        self.filename = {key: stream.filename for key, stream in self.data.items()}

        return sample

    def to_btfeed(self):
        return {key: stream.to_btfeed() for key, stream in self.data.items()}




