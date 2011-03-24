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
    # First layer = PCA-8
    layer1 = {'name' : '1st-PCA',
              'num_components': 8,
              'min_variance': 0,
              'whiten': True,
              }
    
    # Second layer = CAE-6
    layer2 = {'name' : '2nd-CAE',
              'nhid': 6,
              'tied_weights': True,
              'act_enc': 'sigmoid',
              'act_dec': None,
              'irange': 0.001,
              'cost_class' : 'MeanSquaredError',
              'autoenc_class': 'ContractingAutoencoder',
              'corruption_class' : 'BinomialCorruptor',
              'contracting_penalty' : 0.2,
              # Training properties
              'base_lr': 0.001,
              'anneal_start': 100,
              'batch_size' : 20,
              'epochs' : 5,
              }
    
    # Third layer = CAE-6
    layer3 = {'name' : '3rd-CAE',
              'nhid': 6,
              'tied_weights': True,
              'act_enc': 'sigmoid',
              'act_dec': None,
              'irange': 0.001,
              'cost_class' : 'MeanSquaredError',
              'autoenc_class': 'ContractingAutoencoder',
              'corruption_class' : 'BinomialCorruptor',
              'contracting_penalty' : 0.2,
              # Training properties
              'base_lr': 0.001,
              'anneal_start': 100,
              'batch_size' : 20,
              'epochs' : 5,
              }
    
    # Fourth layer = PCA-3
    layer4 = {'name' : '3st-PCA',
              'num_components': 3,
              'min_variance': 0,
              'whiten': False,
              }
    
    # Experiment specific arguments
    conf = {'dataset' : 'sylvester',
            'expname' : 'phase1', # Used to create the submission file
            'transfer' : False,
            'normalize' : True, # (Default = True)
            'normalize_on_the_fly' : False, # (Default = False)
            'randomize_valid' : True, # (Default = True)
            'randomize_test' : True, # (Default = True)
            'resulting_alc' : True, # (Default = False)
            'savedir' : os.path.join(basedir, 'sylvester'),
            'proba' : [1,0,0],
            }

    # Load the dataset
    data = utils.load_data(conf)
    
    # First layer : PCA
    pca1 = create_pca(conf, layer1, data, model=layer1['name'])
    data = [utils.sharedX(pca1.function()(set.get_value(borrow=True)),
                          borrow=True) for set in data]
    
    # Second layer : CAE
    ae1 = create_ae(conf, layer2, data, model=layer2['name'])
    data = [utils.sharedX(ae1.function()(set.get_value(borrow=True)),
                          borrow=True) for set in data]
    
    # Third layer : CAE
    ae2 = create_ae(conf, layer3, data, model=layer3['name'])
    data = [utils.sharedX(ae2.function()(set.get_value(borrow=True)),
                          borrow=True) for set in data]
    
    # Fourth layer : PCA
    pca2 = create_pca(conf, layer4, data)
    
    # Stack layers and create submission file
    block = StackedBlocks([pca1, ae1, ae2, pca2])
    utils.create_submission(conf, block.function())
