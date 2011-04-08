# Standard library imports
import sys
import os

# Third-party imports
from scipy.sparse.csr import csr_matrix

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
from framework.pca import PCA
from framework.base import StackedBlocks
from framework.scripts.experiment import create_pca, create_ae

def create_sparse_pca(conf, layer, data, model=None):
    """
    Simple wrapper to either load a PCA or train it and save its parameters
    """
    savedir = utils.getboth(layer, conf, 'savedir')
    clsname = layer.get('pca_class', 'CovEigPCA')

    # Guess the filename
    if model is not None:
        if model.endswith('.pkl'):
            filename = os.path.join(savedir, model)
        else:
            filename = os.path.join(savedir, model + '.pkl')
    else:
        filename = os.path.join(savedir, layer['name'] + '.pkl')

    # Try to load the model
    if model is not None:
        print '... loading layer:', clsname
        try:
            pca = PCA.load(filename)
        except Exception, e:
            print 'Warning: error while loading %s:' % clsname, e.args[0]
            print 'Switching back to training mode.'

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
    print '... training layer:', clsname
    pca.train(data)

    # Save model parameters
    pca.save(filename)
    return pca

if __name__ == "__main__":
    # First layer = PCA-50 no whiten
    layer1 = {
        'name': '1st-PCA',
        'pca_class': 'SparseMatPCA',
        'num_components': 50,
        'min_variance': 0,
        'whiten': False,
    }

    # Second layer: DAE-200
    layer2 = {
        'name': '2nd-DAE',
        'nhid': 200,
        'tied_weights': True,
        'act_enc': 'sigmoid',
        'act_dec': None,
        'irange': 0.001,
        'cost_class' : 'MeanSquaredError',
        'autoenc_class': 'DenoisingAutoencoder',
        'corruption_class': 'BinomialCorruptor',
        'corruption_level': 0.3,
        'base_lr': 0.001,
        'anneal_start': 100,
        'batch_size': 20,
        'epochs': 5,
        'proba': [1, 0, 0],
    }

    # Third layer = PCA-4 no whiten
    layer3 = {
        'name': '3rd-PCA',
        'pca_class': 'CovEigPCA',
        'num_components': 4,
        'min_variance': 0,
        'whiten': True,
        'proba': [1, 0, 0]
    }

    # Experiment specific arguments
    conf = {'dataset': 'terry',
            'sparse': True,
            'expname': 'dummy', # Used to create the submission file
            'transfer': False,
            'normalize': True, # (Default = True)
            'normalize_on_the_fly': False, # (Default = False)
            'randomize_valid': True, # (Default = True)
            'randomize_test': True, # (Default = True)
            'saving_rate': 2, # (Default = 0)
            'savedir': './outputs',
            }

    # Load the dataset.
    data = utils.load_data(conf)
    
    if conf['transfer']:
        # Data for the ALC proxy
        label = data[3]
        data = data[:3]

    # Discard noninformative features.
    d0 = data[0].shape[1]
    nz_feats = utils.nonzero_features(data, all_subsets=True)
    d = nz_feats.shape[0]
    data = [set[:, nz_feats] for set in data]
    print 'Dropped %i of %i features; %i remaining' % (d0 - d, d0, d)

    # First layer: train or load a PCA
    pca1 = create_sparse_pca(conf, layer1, data[0], model=layer1['name'])
    data = [utils.sharedX(pca1.function()(set), borrow=True) for set in data]

    # Second layer: train or load a DAE
    ae = create_ae(conf, layer2, data, model=layer2['name'])
    data = [utils.sharedX(ae.function()(set.get_value(borrow=True)),
        borrow=True) for set in data]

    # Third layer: train or load a PCA
    pca2 = create_pca(conf, layer3, data, model=layer3['name'])
    data = [utils.sharedX(pca2.function()(set.get_value(borrow=True)),
        borrow=True) for set in data]

    # Compute the ALC for example with labels
    if conf['transfer']:
        data, label = utils.filter_labels(data[0], label)
        alc = embed.score(data, label)
        print '... resulting ALC on train is', alc
        conf['train_alc'] = alc

    # Stack both layers and create submission file
    block = StackedBlocks([pca1, ae, pca2])
    utils.create_submission(conf, block.function(sparse_input=True), features=nz_feats)
