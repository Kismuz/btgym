###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin
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
from subprocess import Popen, PIPE
import glob


class BTgymMonitor():
    """
    Tensorboard monitor support.
    """

    def __init__(self, logdir='./btgym_log', subdir='/', purge_previous=False, **kwargs):
        """
        Tensorboard parameters:
        logdir - where to write tb logs to;
        port - localhost webpage addr to look at;
        reload - web page refresh rate.
        Monitor:
        purge_previous - delete previous logs in logdir if found.
        """
        try:
            import tensorflow as tf

        except:
            raise ModuleNotFoundError('BTgymMonitor requires Tensorflow')
        
        self.tf = tf
        self.process = None
        self.logdir = logdir+subdir

        self.purge_previous = purge_previous

        self.summaries = dict()

        # Remove previous log files if opted:
        if self.purge_previous:
            files = glob.glob(self.logdir + '/*')
            p = Popen(['rm', '-R', ] + files, )  # , stdout=PIPE, stderr=PIPE)

        # Prepare writer:
        self.tf.reset_default_graph()
        self.sess = self.tf.Session()
        self.writer = self.tf.summary.FileWriter(self.logdir, graph=self.tf.get_default_graph())
        self.sess.run(self.tf.global_variables_initializer())

        # Tensorboard on/off:
        self.tensorboard = tensorboard(logdir=logdir, **kwargs)

    def add(self, summary_name, scalars=[], images=[], histograms=[]):
        """
        Adds summary to monitor
        :param summary_name: reference name, dtype=str
        :param scalars: list of names for scalar values, dtype= list of str
        :param images: list of names for image values, dtype= list of str
        :param histograms: list of names for histograms (as 3d Tensor), dtype= list of str
        :return: None
        """
        assert type(summary_name) == str
        self.summaries[summary_name] = TBsummary(summary_name, scalars, images, histograms , self.tf)

    def write(self, summary_name, feed_values, step):
        """
        Updates tensorboard summary with provided data
        :param summary_name: summary reference name
        :param feed_values: dictionary of name=value for this summary
        :return: None
        """
        feeder = dict()

        # Assert feed dictionary is ok:
        try:
            for key in self.summaries[summary_name].feed_holder.keys():
                feeder.update({self.summaries[summary_name].feed_holder[key]: feed_values[key]})
                assert key in feed_values

        except:
            raise AssertionError('Inconsistent {} summary feed\ngiven: {}\nexpected: {}\n'.
                                 format(summary_name,
                                        feed_values.keys(),
                                        self.summaries[summary_name].feed_holder.keys()
                                        )
                         )
        evaluated = self.sess.run(self.summaries[summary_name].tf_summary, feed_dict=feeder)
        self.writer.add_summary(summary=evaluated, global_step=step)
        self.writer.flush()

class TBsummary():
    """____"""

    def __init__(self, name, scalars=[], images=[], histograms=[], tf=None):
        """
        Takes lists of names for every value category.
        1D, 2D Histogram supported.
        """

        self.feed_holder = dict()
        self.name = name
        summaries = []

        for entry in scalars:
            assert type(entry) == str
            self.feed_holder[entry] = tf.placeholder(tf.float32)
            summaries += [tf.summary.scalar(entry, self.feed_holder[entry],)]

        for entry in images:
            assert type(entry) == str
            self.feed_holder[entry] = tf.placeholder(tf.uint8, [None, None, None, 3])
            summaries += [tf.summary.image(entry, self.feed_holder[entry], )]

        for entry in histograms:
            assert type(entry) == str
            self.feed_holder[entry] = tf.placeholder(tf.float32,[None,None,None],)
            summaries += [tf.summary.histogram(entry, self.feed_holder[entry], )]

        self.tf_summary = tf.summary.merge(summaries, name=self.name)
        
class tensorboard():
    """
    Utility class to start/stop tensorboard server.
    """
    def __init__(self, logdir='./btgym_log', port=6006, reload=30,):
        """____"""
        self.port = port
        self.logdir = logdir
        self.process = None

        # Compose start command:
        self.start_string = ['tensorboard']

        assert type(logdir) == str
        self.start_string += ['--logdir={}'.format(logdir)]

        assert type(port) == int
        self.start_string += ['--port={}'.format(port)]

        assert type(reload) == int
        self.start_string += ['--reload_interval={}'.format(reload)]

        self.start_string += ['--purge_orphaned_data']

    def open(self):
        """Launches Tensorboard app."""

        # Kill everything on port-to-use:
        p = Popen(['lsof', '-i:{}'.format(self.port), '-t'], stdout=PIPE, stderr=PIPE)
        pid = p.communicate()[0].decode()[:-1]  # retrieving PID

        if pid is not '':
            p = Popen(['kill', pid])  # , stdout=PIPE, stderr=PIPE)

        # Start:
        self.process = Popen(self.start_string)  # , stdout=PIPE, stderr=PIPE)

    def close(self):
        """Closes tensorboard app."""
        if self.process is not None:
            self.process.terminate()
