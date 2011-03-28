"""An example of experiment made with the new library."""
# Standard library imports
import time
import sys
import os

# Third-party imports
import numpy
import theano
from theano import tensor

from auc import embed

# Local imports
try:
    import framework
except ImportError:
    print >> sys.stderr, \
            "Framework couldn't be imported. Make sure you have the " \
            "repository root on your PYTHONPATH (or as your current " \
            "working directory)"
    sys.exit(1)

from auc import embed
from framework import utils
from framework import cost
from framework import corruption
from framework import autoencoder
from framework import rbm

from framework.pca import PCA, CovEigPCA
from framework.utils import BatchIterator
from framework.base import StackedBlocks
from framework.autoencoder import DenoisingAutoencoder, ContractingAutoencoder
from framework.rbm import GaussianBinaryRBM, PersistentCDSampler
from framework.optimizer import SGDOptimizer

from framework.scripts.experiment import create_pca, create_da

import pylearn.datasets.utlc as pdu

def experiment(conf, pca_conf=None, ae_conf=None, pca2_conf=None):
    # Load the dataset
    data = utils.load_data(conf)
    
    if conf['transfer']:
        # Data for the ALC proxy
        label = data[3]
        data = data[:3]

    blocks = []

    if pca_conf != None:
        # First layer : train or load a PCA
        pca1 = create_pca(conf, pca_conf, data, model=layer1['name'])
        
        data = [utils.sharedX(pca1.function()(set.get_value()), borrow=True)
                for set in data]
        blocks.append(pca1)
   
    if ae_conf != None:
        # Second layer : train or load a DAE or CAE
        ae = create_da(conf, ae_conf, data)#, model=layer2['name'])
        
        data = [utils.sharedX(ae.function()(set.get_value()), borrow=True)
                for set in data]
        blocks.append(ae)
    
    if pca2_conf != None:
        # Third layer : train or load a PCA
        pca2 = create_pca(conf, pca2_conf, data)#, model=layer3['name'])

        data = [utils.sharedX(pca2.function()(set.get_value()), borrow=True)
                for set in data]
        blocks.append(pca2)
    
    # Compute the ALC for example with labels
    if conf['transfer']:
        alc_data, alc_label = utils.filter_labels(data[0], label)

        # To speed up alc calculation, retain only a subset of the 
        # labeled examples
        label_idx = alc_label.any(axis=1)
        alc_data = alc_data[:conf['train_alc_nb'],:]
        alc_label = alc_label[:conf['train_alc_nb'],:]

        print 'computing alc on train'
        train_alc = embed.score(alc_data, alc_label)

        print '... resulting ALC on train is ', train_alc
        conf['train_alc'] = train_alc
    
    # Stack both layers and create submission file
    sblocks = StackedBlocks(blocks)
    utils.create_submission(conf, sblocks.function())

    if conf['dataset'] == 'ule':
        # Compute the valid alc
        # Load the dataset, without permuting valid and test
        (dataset_devel, dataset_valid, dataset_test) = \
            pdu.load_ndarray_dataset("ule", normalize=False, transfer=False) 
        (labels_devel, labels_valid, labels_test)  = pdu.load_ndarray_label("ule")

        print 'computing alc on valid'
        valid_alc = embed.score(sblocks.function()(dataset_valid), labels_valid)

        print '... resulting ALC on valid is ', valid_alc
        conf['valid_alc'] = valid_alc



if __name__ == "__main__":
    # First layer = PCA-75 whiten
    layer1 = {'name' : '1st-PCA',
              'num_components': 75,
              'min_variance': 0,
              'whiten': True,
              # Training properties
              'proba' : [1,1,0],
              'savedir' : './outputs',
              }

    # Second layer = CAE-200
    layer2 = {'name' : '2nd-CAE',
              'nhid': 200,
              'tied_weights': True,
              'act_enc': 'sigmoid',
              'act_dec': None,
              'irange': 0.001,
              'cost_class' : 'MeanSquaredError',
              'autoenc_class': 'ContractingAutoencoder',
              'corruption_class' : 'BinomialCorruptor',
              # 'corruption_level' : 0.3, # For DenoisingAutoencoder
              # Training properties
              'base_lr': 0.001,
              'anneal_start': 100,
              'batch_size' : 20,
              'epochs' : 5,
              'proba' : [1,0,0],
              }
    
    # First layer = PCA-3 no whiten
    layer3 = {'name' : '3st-PCA',
              'num_components': 3,
              'min_variance': 0,
              'whiten': False,
              # Training properties
              'proba' : [0,1,0]
              }
    
    # Experiment specific arguments
    conf = {'dataset' : 'ule',
            'expname' : 'dummy', # Used to create the submission file
            'transfer' : True,
            'normalize' : True, # (Default = True)
            'normalize_on_the_fly' : False, # (Default = False)
            'randomize_valid' : True, # (Default = True)
            'randomize_test' : True, # (Default = True)
            'saving_rate': 2, # (Default = 0)
            'savedir' : './outputs',
            'train_alc' : 0,      # Result of the train alc
            'valid_alc' : 0,      # Result of the valid alc
            'train_alc_nb' : 4000 # Number of examples to use for train alc
            }

    # To compute the cartesian product of the iterable arguments
    def product(*args, **kwds):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = map(tuple, args) * kwds.get('repeat', 1)
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

    for i,num_components in enumerate([1,2,3,4,5,10,15,20,25,30,40,50,60,70,80,90,100]):
        expname = 'exp-%04d'%i
        savedir = './' + expname
        exp_conf = conf.copy()
        exp_conf.update({'expname':expname, 'savedir':savedir})
        exp_layer1 = layer1.copy()
        exp_layer1.update({'num_components':num_components, 'savedir':savedir})
        experiment(exp_conf, exp_layer1)
        s = str(exp_conf['train_alc']) + ' ' +\
            str(exp_conf['valid_alc']) + ' ' + str(num_components)
        print 'Saving ' + s
        f = open('ule_log.txt', 'a')
        f.write(s + '\n')
        f.close()
        

