# Standard library imports
import sys
import os

# Third-party imports
import numpy

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
from utils.alc_corr import class_combs
from framework import utils
from framework.pca import PCA
from framework.base import StackedBlocks

def create_sparse_pca(conf, layer, data):
    """
    Simple wrapper to either load a PCA or train it and save its parameters
    """
    savedir = utils.getboth(layer, conf, 'savedir')
    clsname = layer.get('pca_class', 'SparseMatPCA')

    # Guess the filename.
    filename = os.path.join(savedir, layer['name'])
    if not filename.endswith('.pkl'):
        filename += '.pkl'

    # Try to load the model.
    print '... loading layer:', layer['name']
    try:
        pca = PCA.load(filename)
    except Exception, e:
        print 'Warning: error while loading %s:' % layer['name'], e.args[0]
        print 'Switching back to training mode.'
    else:
        if 'num_components' in layer:
            pca.num_components = layer['num_components']
        if 'min_variance' in layer:
            pca.min_variance = layer['min_variance']
        if 'whiten' in layer:
            pca.whiten = layer['whiten']
        return pca

    MyPCA = framework.pca.get(clsname)
    pca = MyPCA.fromdict(layer)

    # Train the model
    print '... training layer:', layer['name']
    pca.train(data)

    # Save model parameters
    pca.save(filename)

    return pca

if __name__ == "__main__":
    layer1 = {
        'pca_class': 'SparseMatPCA',
        'minibatch_size': 10000,
    }

    conf = {
        'dataset': 'terry',
        'sparse': True,
        'transfer': True,
        'normalize': True,
        'normalize_on_the_fly': False,
        'randomize_valid': True,
        'randomize_test': True,
        'saving_rate': 2,
        'savedir': './outputs'
    }

    # Load the dataset.
    data = utils.load_data(conf)

    # Extract the labels.
    data, labels = data[:3], data[3]

    # Discard noninformative features.
    d0 = data[0].shape[1]
    nz_feats = utils.nonzero_features(data, all_subsets=True)
    d = nz_feats.shape[0]
    data = [set[:, nz_feats] for set in data]
    print 'Dropped %i of %i features; %i remaining' % (d0 - d, d0, d)

    # Set some options for the experiment.
    numpy.random.seed(0xcafebeef)
    fname = 'alc.out' # where to save results
    n_sample = 4096 # how many random examples to use for each ALC computation
    num_reps = 4 # how many times to compute ALC on random sample
    nums_components = range(5000, 50, -50) # list of values for PCA's num_components hyperparam

    # In the initial PCA training, don't keep any more than we'll be needing.
    layer1['num_components'] = max(nums_components)
    # Build an ALC matrix, with a col. for each hyperparam value, and and a row
    # for each sampling.
    alcs = numpy.empty((num_reps * len(list(class_combs(labels.shape[1]))),
        len(nums_components)))

    # Train a PCA with examples labeled with each class subset of size > 1.
    for classes_idx, classes in enumerate(class_combs(labels.shape[1])):
        labeled_train, labeled_labels = utils.filter_labels(data[0], labels, classes=classes)
        print 'Training with %i examples from classes %s' \
            % (labeled_labels.shape[0], str(classes))

        layer1['name'] = 'terry-pca-' + '_'.join(map(str, classes))
        pca = create_sparse_pca(conf, layer1, labeled_train)

        # This is the internal loop so that we don't have to retrain for different
        # num_components values.
        for (k_idx, k) in enumerate(nums_components):
            pca.num_components = k

            for i in range(num_reps):
                # Get a random sampling of transformed examples from this subset.
                rand_idx = sorted(numpy.random.permutation(labeled_train.shape[0])[:n_sample])
                train_pca = pca.function()(labeled_train[rand_idx])
                alc = embed.score(train_pca, labeled_labels[rand_idx])

                print classes_idx, k_idx, i, alc
                alcs[i + num_reps * classes_idx, k_idx] = alc

    # Save the ALC matrix to a file.
    if fname is not None:
        numpy.savetxt(fname, alcs, fmt='%.4f')
        print 'ALCs saved in %s' % fname
