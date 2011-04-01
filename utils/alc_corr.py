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

import pylearn.datasets.utlc as pdu

def create_pca(conf, layer, data, model=None):
    """
    Simple wrapper to either load a PCA or train it and save its parameters
    """
    #savedir = utils.getboth(layer, conf, 'savedir')
    #if model is not None:
    #    # Load the model
    #    if not model.endswith('.pkl'):
    #        model += '.pkl'
    #    try:
    #        print '... loading PCA layer'
    #        filename = os.path.join(savedir, model)
    #        return PCA.load(filename)
    #    except Exception, e:
    #        print 'Warning: error while loading PCA.', e.args[0]
    #        print 'Switching back to training mode.'

    # Train the model
    MyPCA = framework.pca.get(layer.get('pca_class', 'CovEigPCA'))
    pca = MyPCA.fromdict(layer)
    pca.train(data)

    return pca

def experiment(conf, pca_conf, data, label):
    blocks = []

    # load and train the model
    pca = create_pca(conf, pca_conf, data, model=layer1['name'])
    data = pca.function()(data)
    blocks.append(pca)
    sblocks = StackedBlocks(blocks)

    alc = embed.score(data, label)
    conf['alc'] = alc


if __name__ == "__main__":
    # First layer = PCA-75 whiten
    layer1 = {'name' : 'PCA',
              'num_components': 75,
              'min_variance': 0,
              'whiten': True,
              # Training properties
              'proba' : [1,1,0],
              'savedir' : './outputs',
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
            'train_alc_nb' : 4096 # Number of examples to use for train alc
            }


    dataset = utils.load_data(conf)

    for i,num_components in enumerate([2,3,4,5,10,15,20,25,30,50,70,90,100]):
        print "pca-" + str(num_components)
        expname = 'exp-%04d'%i
        savedir = './' + expname
        exp_conf = conf.copy()
        exp_conf.update({'expname':expname, 'savedir':savedir})

        exp_layer1 = layer1.copy()
        exp_layer1.update({'num_components':num_components, 'savedir':savedir})

        _ , dataset_valid, _ = \
            pdu.load_ndarray_dataset("ule", normalize=False, transfer=False)
        _ , labels_valid, _ = pdu.load_ndarray_label("ule")

        experiment(exp_conf, exp_layer1, dataset_valid, labels_valid)
        s = 'valid ' + str(exp_conf['alc'])
        print s
        f = open(expname+ '.txt', 'a')
        f.write(s + '\n')
        f.close()

        # train on valid
        data = dataset[0].get_value(borrow=True)
        label = dataset[3].get_value(borrow=True)

        numpy.random.seed(0xcafebeef)

        # try with 2 indexes from transfer
        idx_2 = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        for idx in idx_2:
            label_idx = (label[:,idx] > 0).any(axis=1)

            tmp_data  = data[label_idx]
            tmp_label = label[:,idx]
            tmp_label = tmp_label[label_idx]

            for i in range(4):
                rand_idx = sorted(numpy.random.permutation(tmp_data.shape[0])[:4096])
                experiment(exp_conf, exp_layer1, tmp_data[rand_idx], tmp_label[rand_idx])
                s = str(idx) + ' ' + str(exp_conf['alc'])
                print s
                f = open(expname+ '.txt', 'a')
                f.write(s + '\n')
                f.close()

        # try with 3 indexes from transfer
        idx_3 = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
        # try it for 3 idx on train
        for idx in idx_3:
            label_idx = (label[:,idx] > 0).any(axis=1)

            tmp_data  = data[label_idx]
            tmp_label = label[:,idx]
            tmp_label = tmp_label[label_idx]

            for i in range(8):
                rand_idx = sorted(numpy.random.permutation(tmp_data.shape[0])[:4096])
                experiment(exp_conf, exp_layer1, tmp_data[rand_idx], tmp_label[rand_idx])
                s = str(idx) + ' ' + str(exp_conf['alc'])
                print s
                f = open(expname+ '.txt', 'a')
                f.write(s + '\n')
                f.close()

