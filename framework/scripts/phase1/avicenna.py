"""An example of experiment made with the new library."""
# Standard library imports
import sys
import os
import getpass

# Local imports
try:
    import framework
except ImportError:
    print >> sys.stderr, \
            "Framework couldn't be imported. Make sure you have the " \
            "repository root on your PYTHONPATH (or as your current " \
            "working directory)"
    sys.exit(1)

from framework import utils
from framework.base import StackedBlocks
from framework.scripts.experiment import create_pca, create_ae

##################################################
basedir = os.path.join('/data/lisatmp/ift6266h11/', getpass.getuser(), 'phase1')

if __name__ == "__main__":
    # First layer = PCA-75 whiten
    layer1 = {'name' : '1st-PCA',
              'num_components': 75,
              'min_variance': 0,
              'whiten': True,
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
              'contracting_penalty' : 0.2, # For ContractingAutoencoder
              # Training properties
              'base_lr': 0.001,
              'anneal_start': 100,
              'batch_size' : 20,
              'epochs' : 50,
              }

    # First layer = Transductive PCA-3 whiten
    layer3 = {'name' : '3st-PCA',
              'num_components': 3,
              'min_variance': 0,
              'whiten': True,
              }

    # Experiment specific arguments
    conf = {'dataset' : 'avicenna',
            'expname' : 'dummy', # Used to create the submission file
            'transfer' : False,
            'normalize' : True, # (Default = True)
            'normalize_on_the_fly' : False, # (Default = False)
            'randomize_valid' : True, # (Default = True)
            'randomize_test' : True, # (Default = True)
            'savedir' : os.path.join(basedir, 'avicenna'),
            'proba' : [1, 0, 0],
            }

    # Load the dataset
    data = utils.load_data(conf)

    # First layer : PCA
    pca1 = create_pca(conf, layer1, data)
    data = [utils.sharedX(pca1.function()(set.get_value()), borrow=True)
            for set in data]

    # Second layer : CAE
    ae = create_ae(conf, layer2, data)
    data = [utils.sharedX(ae.function()(set.get_value()), borrow=True)
            for set in data]

    # Third layer : Transductive PCA
    pca2 = []
    for p in ([0, 1, 0], [0, 0, 1]):
        layer3.update(proba=p)
        pca2.append(create_pca(conf, layer3, data))

    # Stack layers and create submission file
    block = [StackedBlocks([pca1, ae, pca]) for pca in pca2]
    utils.create_submission(conf, block[0].function(), block[1].function())
