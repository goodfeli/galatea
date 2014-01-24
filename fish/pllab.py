#! /usr/bin/env ipythonpl

'''
Pylearn2 Lab.
By: Jason Yosinski
'''

import pdb
import os
import gc
from pylab import *
from numpy import *
from IPython import embed
from PIL import Image
import IPython

# Standard library imports
import argparse
import gc
import logging
import os

# Third-party imports
import numpy as np

# Local imports
from pylearn2.utils import serial
from pylearn2.utils.logger import (
    CustomStreamHandler, CustomFormatter, restore_defaults
)
from pylearn2.config import yaml_parse

from pylearn2.utils.data_specs import DataSpecsMapping    
from pylearn2.space import CompositeSpace
from pylearn2.utils.iteration import is_stochastic
from fish.dl_util.plotting import tile_raster_images, tile_raster_images_tensor, histPlus, imagesc, tile_shape

# IPython
# Reference: http://ipython.org/ipython-doc/dev/interactive/reference.html#embedding-ipython
from IPython.terminal.embed import InteractiveShellEmbed
from IPython.config.loader import Config
cfg = Config()
cfg.TerminalInteractiveShell.confirm_exit = False
ipshell = InteractiveShellEmbed(config=cfg,
                                banner1 = 'Dropping into IPython session. Poke around, plot, and then push Ctrl-D to exit.',
                                exit_msg = 'Exited interpreter.')


# Check X display up front
if os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"') != 0:
    raise Exception('X not working %s' % ('(DISPLAY=%s)' % os.environ['DISPLAY'] if 'DISPLAY' in os.environ else '(DISPLAY not set)'))



def main():
    parser = argparse.ArgumentParser(description='Pylearn2 lab.')
    parser.add_argument('-s', '--save', action='store_true', help = 'Save the resulting images')
    parser.add_argument('-q', '--quit', action='store_true', help = 'Quit after plotting instead of dropping into IPython')
    parser.add_argument('directory', type = str,
                        help = 'Which results directory to use')
    args = parser.parse_args()

    # OLD
    #config_file_path = '/home/jason/s/deep_learning/pylearn/pred_net.yaml'
    #train = yaml_parse.load_path(config_file_path)
    #train = serial.load_train_file(config_file_path)

    #result_prefix = '/home/jason/s/pylearn2/pylearn2/pred/results/'
    result_prefix = '/u/yosinski/s/galatea/fish/results/'
    result_dir = os.path.join(result_prefix, args.directory)

    print 'loading train object...'
    #train = serial.load_train_file(os.path.join(result_dir, 'pred_net.yaml'))
    train = serial.load_train_file(os.path.join(result_dir, 'model.yaml'))
    print 'loading saved model...'
    #model = serial.load(os.path.join(result_dir, 'pred_net.pkl'))
    model = serial.load(os.path.join(result_dir, 'model.pkl'))
    print 'done.'

    print 'model was trained on:'
    print model.dataset_yaml_src

    if train.algorithm.cost is not None:
        data_specs = train.algorithm.cost.get_data_specs(model)
    else:
        data_specs = train.model.get_default_cost().get_data_specs(train.model)
    mapping = DataSpecsMapping(data_specs)
    space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
    source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
    flat_data_specs = (CompositeSpace(space_tuple), source_tuple)

    num_frames = model.num_frames
    num_batches = 100
    batch_size = train.algorithm.batch_size if train.algorithm.batch_size else 20*num_frames
    
    train_dataset = train.dataset
    valid_dataset = train.algorithm.monitoring_dataset['valid']
    
    rng = train.algorithm.rng
    if not is_stochastic(train.algorithm.train_iteration_mode):
        rng = None
    
    train_iterator = train_dataset.iterator(mode = train.algorithm.train_iteration_mode,
                                            batch_size = batch_size,
                                            data_specs = flat_data_specs,
                                            return_tuple = True, rng=rng,
                                            num_batches = num_batches * 10)
    valid_iterator = valid_dataset.iterator(mode = train.algorithm.train_iteration_mode,
                                            batch_size = batch_size,
                                            data_specs = flat_data_specs,
                                            return_tuple = True,  # No rng override
                                            num_batches = num_batches * 10)

    train_batches = [train_iterator.next() for ii in range(num_batches)]
    valid_batches = [valid_iterator.next() for ii in range(num_batches)]

    print 'got batches with shape:'
    for dat in train_batches[0]:
        print '  ', dat.shape



    #########################
    # Plot costs
    #########################

    # Plot costs over time
    ch_train_objective = model.monitor.channels['train_objective']
    ch_valid_objective = model.monitor.channels['valid_objective']

    x_vals = ch_train_objective.epoch_record
    x_label = 'epoch'

    
    plot(x_vals, ch_train_objective.val_record, 'b-')
    plot(x_vals, ch_valid_objective.val_record, 'r-')
    legend(('train', 'valid'))

    if args.save:
        savefig(os.path.join(result_dir, 'costs_lin.png'))
        savefig(os.path.join(result_dir, 'costs_lin.pdf'))
    if args.save:
        gca().set_yscale('log')
        savefig(os.path.join(result_dir, 'costs_log.png'))
        savefig(os.path.join(result_dir, 'costs_log.pdf'))
        gca().set_yscale('linear')



        

    #########################
    # Compute some accuracies
    #########################

    n_correct = 0
    n_total = 0
    for bb,batch in enumerate(train_batches):
        feat,ids,xy = batch
        ids_hat,xy_hat = model.fns.feat_to_idxy(feat)

        idx_true = np.where( ids == 1 )[1]
        idx_hat = np.where(np.sign(ids_hat.T - ids_hat.max(1)).T + 1)[1]
        n_correct += (idx_true == idx_hat).sum()
        n_total += len(idx_true)
    print 'Class ID accuracy:', float(n_correct)/n_total

        
        #batch_size = ids.shape[0]
        #for ii in xrange(batch_size):
        #    idx_true = np.where(ids[ii,:])[0][0]
        #    print ii, idx_true, ids_hat[ii,idx_true] / ids_hat[ii,:].max()
        
        
    #########################
    # Embed
    #########################

    if not args.quit:
        # Start shell
        ipshell()
    print 'done.'



if __name__ == '__main__':
    main()
