import os

import pandas as pd
from logbook import Logger, INFO


class ParseHistDataComConfig:
    sep = ';',
    header = 0,
    index_col = 0,
    parse_dates = True,
    names = ['open', 'high', 'low', 'close', 'volume']


class ForexReader:
    def __init__(self, data_filename: str) -> None:
        self.log = Logger("ForexReader", level=INFO)
        self.parsing_params = ParseHistDataComConfig()
        self._read_csv(data_filename)

    def _read_csv(self, data_filename=None):
        """
        Populates instance by loading data: CSV file --> pandas dataframe.

        Args:
            data_filename: [opt] csv data filename as string or list of such strings.
        """

        self.filename = data_filename
        if type(self.filename) == str:
            self.filename = [self.filename]

        dataframes = []
        for filename in self.filename:
            try:
                assert filename and os.path.isfile(filename)
                current_dataframe = pd.read_csv(
                    filename,
                    sep=self.parsing_params.sep,
                    header=self.parsing_params.header,
                    index_col=self.parsing_params.index_col,
                    parse_dates=self.parsing_params.parse_dates,
                    names=self.parsing_params.names,
                )

                # Check and remove duplicate datetime indexes:
                duplicates = current_dataframe.index.duplicated(keep='first')
                how_bad = duplicates.sum()
                if how_bad > 0:
                    current_dataframe = current_dataframe[~duplicates]
                    self.log.warning('Found {} duplicated date_time records in <{}>.\
                     Removed all but first occurrences.'.format(how_bad, filename))

                dataframes += [current_dataframe]
                self.log.info('Loaded {} records from <{}>.'.format(dataframes[-1].shape[0], filename))

            except:
                msg = 'Data file <{}> not specified / not found / parser error.'.format(str(filename))
                self.log.error(msg)
                raise FileNotFoundError(msg)

        self.data = pd.concat(dataframes)
        data_range = pd.to_datetime(self.data.index)
        self.total_num_records = self.data.shape[0]
        self.data_range_delta = (data_range[-1] - data_range[0]).to_pytimedelta()

        def get_data():
            pass

        def get_data_range_delta():
            pass
