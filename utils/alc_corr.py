"""An example of experiment made with the new library."""
# Standard library imports
import time
import sys
import os
import pprint

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

from framework.scripts.experiment import create_pca, create_ae

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

def create_model(conf, layer_conf, data):
    if layer_conf['type'] == 'PCA':
        return create_pca(conf, layer_conf, data)
    elif layer_conf['type'] == 'AE':
        return create_ae(conf, layer_conf, data)

def experiment(conf, layer_confs, train_data, alc_data, alc_label):
    blocks = []
    data = train_data

    # load and train the model
    for layer_conf in layer_confs:
        model = create_model(conf, layer_conf, data)
        data  = model.function()(data)
        blocks.append(model)

    
    sblocks = StackedBlocks(blocks)
    alc = embed.score(sblocks.function()(alc_data), alc_label)
    conf['alc'] = alc

# Generator to iterate on all combinations
def comb(n,k):
    def gen(l,h,k):
        if k == 1:
            for i in range(l,h):
                yield (i,)
        else:
            for i in range(l,h-k+1):
                for j in gen(i+1,h,k-1):
                    yield (i,) + j

    for i in gen(0,n,k):
        yield i

def class_combs(n):
    for k in range(2,n+1):
        for i in comb(n, k):
            yield i

def option_iter(options):
    def gen(options, keys, i):
        if i == len(keys):
            yield {}
        else:
            k = keys[i]
            for e in options[k]:
                for r in gen(options, keys, i+1):
                    r.update({k:e})
                    yield r

    for option in gen(options, options.keys(), 0):
        yield option

def layer_iter(layers):
    def gen(layers, i):
        if i == len(layers):
            yield ()
        else:
            for o in option_iter(layers[i]):
                for r in gen(layers, i+1):
                    yield (o,) + r
    for layers_options in gen(layers,0):
        yield layers_options

def confs_iter(confs):
    for layers_options in layer_iter(confs['layers_options']):
        conf = confs['conf']
        layers = [layer.copy() for layer in confs['layers']]

        for i, layer in enumerate(layers):
            layer.update(layers_options[i])

        yield (conf, layers)

def append_result(filename, row_list, verbose=False, sep=' '): 
    f = open(filename, 'a')
    s = sep.join(map(str, row_list))
    if verbose:
        print s
    f.write(s+'\n')
    f.close()

def append_log(filename, s, verbose=False):
    f = open(filename, 'a')
    if verbose:
        print s
    f.write(s + '\n')
    f.close()

def append_log_header(filename, confs_file, confs, min_exp_idx, max_exp_idx, transfer_labels_nb, verbose):
    s = "Computing alc values using configurations defined in '" + confs_file + "':\n"
    s += pprint.pformat(confs)
    s += "\n"
    s += "for exp_idx from " + str(min_exp_idx) + " to " + str(max_exp_idx) + "\n"
    s += "\n"

    s +=     "exp_idx options\n"
    s +=     "------- -------\n"
    # All exp_idx options
    for exp_idx, options in enumerate(layer_iter(confs['layers_options'])):
        s += "%04d    %s\n"%(exp_idx, pprint.pformat(options))

    s += "\n"

    s +=     "transfer_cls_idx transfer classes selected\n"
    s +=     "---------------- ------------------------\n"
    # All transfer classes indexes and their corresponding classes
    for transfer_cls_idx, idx in enumerate(class_combs(transfer_labels_nb)):
        s += "%04d             %s\n"%(transfer_cls_idx, pprint.pformat(idx))
    
    s += "\n"

    s += "Result format\n"
    s += "exp_idx train_cls_idx perm_idx train_alc valid_alc test_alc\n"

    s += "----------------- Execution log ------------------------------"

    append_log(filename, s, verbose)
    

def randomize_labels(dataset):
    # HACK because pylearn does not allow randomizing the labels
    rng     = numpy.random.RandomState([1,2,3,4])
    perm    = rng.permutation(dataset.shape[0])
    return dataset[perm]

def default_confs():
    return {
        'layers': [
            # First layer = PCA-75 whiten
            {'name' : 'PCA',
             'type' : 'PCA',
             'num_components': 75,
             'min_variance': 0,
             'whiten': True,
             # Training properties              
             'proba' : [1,0,0],
             'savedir' : './outputs',
             },
        ],

        # Experiment specific arguments
        'conf':{'dataset' : 'ule',
                'expname' : 'dummy', # Used to create the submission file
                'transfer' : True,
                'normalize' : True, # (Default = True)
                'normalize_on_the_fly' : False, # (Default = False)
                'randomize_valid' : False, # (Default = True)
                'randomize_test' : False, # (Default = True)
                'saving_rate': 2, # (Default = 0)
                'savedir' : './outputs',
                'train_alc' : 0,      # Result of the train alc
                'valid_alc' : 0,      # Result of the valid alc
                'train_alc_nb' : 4096, # Number of examples to use for train alc
                'alc_perm_nb' : 4,        # Number of permutations of example to compute
                },

        # Parameters to try for the different layers
        'layers_options': [
            {'num_components':[2,3,4,5,10,15,20,25,30,50,70,90,100]},
        ],
    }


if __name__ == "__main__":

    args    = sys.argv
    options = [ a for a in args if a.find("-") != -1 ]
    args    = [ a for a in args if a.find("-") == -1 ]

    # Load configuration dictionaries
    if len(args) >= 2:
        confs_file = args[1]
        confs = eval(open(confs_file, 'r').read())
    else:
        confs_file = 'default.conf'
        confs = default_confs()

    if len(args) >= 4:
        min_exp_idx = int(args[2])
        max_exp_idx = int(args[3])
    else:
        min_exp_idx = 0
        max_exp_idx = None

    if '-v' in options or '--verbose' in options:
        verbose = True
    else:
        verbose = False

    conf = confs['conf']

    confs_name = confs_file[:-5]
    filename = 'results_' + confs_name + '_' + str(min_exp_idx) + '_' + str(max_exp_idx)
    results = filename + '.txt'
    log = filename + '.log'

    if os.path.exists(results):
        os.remove(results)

    if os.path.exists(log):
        os.remove(log)

    dataset = utils.load_data(conf)
    dataset_train = dataset[0].get_value(borrow=True)
    transfer_labels = dataset[3].get_value(borrow=True)

    append_log_header(log, confs_file, confs, min_exp_idx, max_exp_idx, transfer_labels.shape[1], verbose)

    for exp_idx,(conf, layers) in enumerate(confs_iter(confs)):

        # Skip computing experiments that were not requested
        if exp_idx < min_exp_idx:
            append_log(log, "skipping " + str(exp_idx), verbose)
            continue

        if max_exp_idx != None and exp_idx >= max_exp_idx:
            append_log(log, "stopping before " + str(exp_idx), verbose)
            break

        # Compute valid and test alc only on ule
        if conf['dataset'] == 'ule':
            dataset_valid = dataset[1].get_value(borrow=True)
            dataset_test  = dataset[2].get_value(borrow=True)
            _ , labels_valid, labels_test = pdu.load_ndarray_label("ule")

            if conf['randomize_valid']:
                labels_valid = randomize_labels(labels_valid)
            if conf['randomize_test']:
                labels_test  = randomize_labels(labels_test)

            experiment(conf, layers, dataset_valid, dataset_valid, labels_valid)
            valid_alc = conf['alc']

            experiment(conf, layers, dataset_test, dataset_test, labels_test)
            test_alc = conf['alc']

            append_log(\
                log,\
                "valid alc: " + str(valid_alc) + " test alc: " + str(test_alc),\
                verbose)
        else:
            valid_alc = 0
            test_alc = 0


        numpy.random.seed(0xcafebeef)

        # for each class combination (i.e. (0,1), (1,2), ..., (0, 1, 2), etc)
        for train_cls_idx, idx in enumerate(class_combs(transfer_labels.shape[1])):
            label_idx = (transfer_labels[:,idx] > 0).any(axis=1)

            tmp_data  = dataset_train[label_idx]
            tmp_label = transfer_labels[:,idx]
            tmp_label = tmp_label[label_idx]

            # repeat for different examples
            for perm_idx in range(4):
                rand_idx = sorted(numpy.random.permutation(tmp_data.shape[0])[:4096])

                # compute the alc
                experiment(conf, layers, tmp_data[rand_idx], tmp_data[rand_idx], tmp_label[rand_idx])
                append_result(results, [exp_idx, train_cls_idx, perm_idx, conf['alc'], valid_alc, test_alc], verbose)
