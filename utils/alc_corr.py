"""An example of experiment made with the new library."""
# Standard library imports
import time
import sys
import os
import pprint
import itertools

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
from framework.utils import utlc as utils
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

def create_transductive_pca(conf, layer, data, model=None):
    """
    data should be a numpy array
    """
    # Train the model
    MyPCA = framework.pca.get(layer.get('pca_class', 'CovEigPCA'))
    pca = MyPCA.fromdict(layer)
    pca.train(data)

    return (pca.function()(data), pca)

def create_model(conf, layer_conf, data):
    if layer_conf['type'] == 'PCA':
        return create_pca(conf, layer_conf, data)
    elif layer_conf['type'] == 'AE':
        return create_ae(conf, layer_conf, data)

def train(conf, layer_confs, train_data):
    blocks = []
    data = train_data

    # load and train the model
    for layer_conf in layer_confs:
        model = create_model(conf, layer_conf, data)
        data = [utils.sharedX(model.function()(set.get_value(borrow=True)),
                              borrow=True) for set in data]
        blocks.append(model)

    return data, blocks

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
    for exp_idx, options in enumerate(itertools.chain(['raw'], layer_iter(confs['layers_options']))):
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

def run_experiments(confs, results, log, min_exp_idx, max_exp_idx, verbose):
    conf = confs['conf']

    dataset = utils.load_data(conf)
    transfer_labels = dataset[3].get_value(borrow=True)
    dataset = dataset[:3]

    append_log_header(log, confs_file, confs, min_exp_idx, max_exp_idx, transfer_labels.shape[1], verbose)

    label_idx = transfer_labels.any(axis=1)

    for exp_idx,(conf, layers) in enumerate(itertools.chain(((confs['conf'], []),), confs_iter(confs))):

        # Skip computing experiments that were not requested
        if exp_idx < min_exp_idx:
            append_log(log, "skipping " + str(exp_idx), verbose)
            continue

        if max_exp_idx != None and exp_idx >= max_exp_idx:
            append_log(log, "stopping before " + str(exp_idx), verbose)
            break

        non_transductive = [ layer for layer in layers if not layer.get('transductive', False) ]
        transductive = [ layer for layer in layers if layer.get('transductive', False) ]

        assert len(transductive) <= 1, "More than 1 transductive layer given"

        if len(transductive) == 1:
            assert transductive[0].get("type", False) == "PCA", "Transductive layer is not a PCA"

        new_reps, blocks = train(conf, non_transductive, dataset)

        # Compute valid and test alc only on ule
        if conf['dataset'] == 'ule':
            _ , labels_valid, labels_test = pdu.load_ndarray_label("ule")

            if conf['randomize_valid']:
                labels_valid = randomize_labels(labels_valid)
            if conf['randomize_test']:
                labels_test  = randomize_labels(labels_test)

            if len(transductive) == 1:
                newrep_valid, _ = create_transductive_pca(conf, transductive[0], new_reps[1].get_value(borrow=True))
            else:
                newrep_valid = new_reps[1].get_value(borrow=True)

            valid_alc = embed.score(newrep_valid, labels_valid)

            if len(transductive) == 1:
                newrep_test, _ = create_transductive_pca(conf, transductive[0], new_reps[2].get_value(borrow=True))
            else:
                newrep_test = new_reps[2].get_value(borrow=True)

            test_alc = embed.score(newrep_test, labels_test)

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

            tmp_data  = new_reps[0].get_value(borrow=True)[label_idx]
            tmp_label = transfer_labels[:,idx]
            tmp_label = tmp_label[label_idx]

            # repeat for different examples
            for perm_idx in range(conf['alc_perm_nb']):
                rand_idx = sorted(numpy.random.permutation(tmp_data.shape[0])[:conf['train_alc_nb']])

                if len(transductive) == 1:
                    newrep_train, _ = create_transductive_pca(conf, transductive[0], tmp_data[rand_idx])
                else:
                    newrep_train = StackedBlocks(blocks).function()(tmp_data[rand_idx])
                    
                train_alc = embed.score(newrep_train, tmp_label[rand_idx])

                # compute the alc
                append_result(results, [exp_idx, train_cls_idx, perm_idx, train_alc, valid_alc, test_alc], verbose)

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

    confs_name = os.path.splitext(os.path.basename(confs_file))[0]
    filename = 'results_' + confs_name + '_' + str(min_exp_idx) + '_' + str(max_exp_idx)
    results = filename + '.txt'
    log = filename + '.log'

    if os.path.exists(results):
        os.remove(results)

    if os.path.exists(log):
        os.remove(log)

    run_experiments(confs, results, log, min_exp_idx, max_exp_idx, verbose)
