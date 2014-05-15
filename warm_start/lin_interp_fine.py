#!/usr/bin/env python
"""
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
# Standard library imports
import argparse
import logging

# Third-party imports
import numpy as np

# Local imports
from pylearn2.utils import serial
from pylearn2.utils.logger import (
    CustomStreamHandler, CustomFormatter, restore_defaults
)


class FeatureDump(object):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    encoder : WRITEME
    dataset : WRITEME
    path : WRITEME
    batch_size : WRITEME
    topo : WRITEME
    """
    def __init__(self, encoder, dataset, path, batch_size=None, topo=False):
        self.encoder = encoder
        self.dataset = dataset
        self.path = path
        self.batch_size = batch_size
        self.topo = topo

    def main_loop(self, **kwargs):
        """
        .. todo::

            WRITEME
        """
        if self.batch_size is None:
            if self.topo:
                data = self.dataset.get_topological_view()
            else:
                data = self.dataset.get_design_matrix()
            output = self.encoder.perform(data)
        else:
            myiterator = self.dataset.iterator(mode='sequential',
                                               batch_size=self.batch_size,
                                               topo=self.topo)
            chunks = []
            for data in myiterator:
                chunks.append(self.encoder.perform(data))
            output = np.concatenate(chunks)
        np.save(self.path, output)


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Launch an experiment from a YAML configuration file.",
        epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--level-name', '-L',
                        action='store_true',
                        help='Display the log level (e.g. DEBUG, INFO) '
                             'for each logged message')
    parser.add_argument('--timestamp', '-T',
                        action='store_true',
                        help='Display human-readable timestamps for '
                             'each logged message')
    parser.add_argument('--time-budget', '-t', type=int,
                        help='Time budget in seconds. Stop training at '
                             'the end of an epoch if more than this '
                             'number of seconds has elapsed.')
    parser.add_argument('--verbose-logging', '-V',
                        action='store_true',
                        help='Display timestamp, log level and source '
                             'logger for every logged message '
                             '(implies -T).')
    parser.add_argument('--debug', '-D',
                        action='store_true',
                        help='Display any DEBUG-level log messages, '
                             'suppressed by default.')
    parser.add_argument('config', action='store',
                        choices=None,
                        help='A YAML configuration file specifying the '
                             'training procedure')
    return parser


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    train_obj = serial.load_train_file(args.config)
    try:
        iter(train_obj)
        iterable = True
    except TypeError as e:
        iterable = False

    # Undo our custom logging setup.
    restore_defaults()
    # Set up the root logger with a custom handler that logs stdout for INFO
    # and DEBUG and stderr for WARNING, ERROR, CRITICAL.
    root_logger = logging.getLogger()
    if args.verbose_logging:
        formatter = logging.Formatter(fmt="%(asctime)s %(name)s %(levelname)s "
                                          "%(message)s")
        handler = CustomStreamHandler(formatter=formatter)
    else:
        if args.timestamp:
            prefix = '%(asctime)s '
        else:
            prefix = ''
        formatter = CustomFormatter(prefix=prefix, only_from='pylearn2')
        handler = CustomStreamHandler(formatter=formatter)
    root_logger.addHandler(handler)
    # Set the root logger level.
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)

    if iterable:
        raise NotImplementedError()

    train_obj.setup()
    extensions = train_obj.extensions

    found = False
    for x in extensions:
        if "MonitorBasedSaveBest" in str(type(x)):
            found = True
            break
    assert found

    model_path = x.save_path

    soln = serial.load(model_path).get_param_vector()
    model = train_obj.model
    init = model.get_param_vector()

    num_epochs = 200

    for i in xrange(num_epochs):
        alpha = float(i) / float(num_epochs - 1)
        model.set_param_vector((1 - alpha) * init + alpha * soln)
        model.monitor()
        model.monitor.report_batch(1)
        model.monitor.report_epoch()

    serial.save('interpolated.pkl', model)
