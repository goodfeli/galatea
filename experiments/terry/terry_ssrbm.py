# Third-party imports
import numpy
from pylearn.datasets import utlc
import theano

# Local imports
from auc import embed
from framework.scripts import experiment as exp
from framework import utils


def experiment0(state, channel):

    ## Default values
    ## --------------

    # All the values in the configuration dictionaries can be overridden
    # from state.

    conf = {'dataset_path': '/data/lisatmp/ift6266h11/lamblinp/terry_pca_512.npz',
            # /u/goodfeli/ift6266h11/experiments/fast_sc/data/terry_pca_512.mat
            # gives 3 dictionaries, with keys 'devel', 'valid' and 'test'
            # The *first* row from 'valid' and 'test' should be discarded.
            'expname': 'Zarathustra',
            'transfer': True, # default: True
            'savedir': './outputs',
            }

    layer1 = {
            'name': 'ssRBM_1',
            'rbm_class': 'mu_pooled_ssRBM',
            'seed': 999,

            'nhid': 100, ###
            'n_s_per_h': 1, ##
            'batch_size': 100,

            ## Initialization & constraints
            'alpha0': 5., ###
            'alpha_irange': 0.,
            'alpha_min': 1.,
            'alpha_max': 100.,
            'b0': 0.,
            'B0': 10., ###
            'B_min': 1.0,
            'B_max': 101.0,
            'Lambda0': 0.00001, ###
            'Lambda_min': 0.00001,
            'Lambda_max': 10.,
            'Lambda_irange': 0.,
            'mu0': 1.,
            'W_irange': None, # Will use a sensible default
            'particles_min': -30.,
            'particles_max': +30.,

            ## Sampler
            'sampler': 'PersistentCDSampler',
            'pcd_steps': 1,

            ## Optimizer
            'optimizer': 'SGDOptimizer',
            'base_lr': 1e-2, ###
            #'anneal_start': None,

            # Training
            'epochs': 100,
            'proba': [1, 0, 0],
            'saving_rate': 10,
            }

    trans_pca_valid = {
            'name': 'transPCA_valid',
            'num_components': 4,
            'whiten': False,
            'proba': [0, 1, 0],
            }

    trans_pca_test = {
            'name': 'transPCA_test',
            'num_components': 4,
            'whiten': False,
            'proba': [0, 0, 1],
            }

    # Synchronize default values above with values specified in state
    for dname in ('conf', 'layer1', 'trans_pca_valid', 'trans_pca_test'):
        if dname in state:
            state_keys = set(state[dname].keys())
            local_keys = set(locals()[dname].keys())
            unknown_keys = state_keys - local_keys
            if unknown_keys:
                print >> sys.stderr, ('Warning: The following keys will be '
                        'ignored, because they are invalid.')
                for key in unknown_keys:
                    print >> sys.stderr, ('    state.%s.%s: %s' % (
                        dname, key, state[dname][key]))
            # Update local dict
            locals()[dname].update(state[dname])
        else:
            state[dname] = {}

        # Update state with default values
        state[dname].update(locals()[dname])

    ## Load data
    ## ---------

    if 'dataset_path' in conf:
        if conf['dataset_path'][-4:] == '.mat':
            from scipy.io import loadmat
            data = loadmat(conf['dataset_path'])
        elif conf['dataset_path'][-4:] == '.npz':
            import pdb; pdb.set_trace()
            data = numpy.load(conf['dataset_path'])
        else:
            raise ValueError('Unknown extension: %s' % conf['dataset_path'][-4:])

        data = (data['devel'], data['valid'], data['test'])

        if 'dataset' in conf:
            dataset_name = conf['dataset']
        else:
            for name in ['avicenna','harry','rita','sylvester','terry','ule']:
                if name in conf['dataset_path'][:-4]:
                    dataset_name = name
                    conf['dataset'] = name
                    break
            else:
                # If no dataset name was in there
                raise ValueError('Unable to figure out the name of '
                        'the data set', conf['dataset_path'])

        # TERRY hack
        if dataset_name == 'terry':
            if len(data[1]) == len(data[2]) == 4097:
                data = (data[0], data[1][1:], data[2][1:])
            assert len(data[1]) == len(data[2]) == 4096

        if conf.get('transfer', False):
            label = utlc.load_ndarray_transfer(dataset_name)

    elif 'dataset' in conf:
        data = utils.load_data(conf)
        if conf.get('transfer', False):
            label = data[3]
            data = data[:3]

    else:
        raise ValueError('I have no idea where to find your data '
                'or how to load it.')

    data = [utils.sharedX(dataset, borrow=True) for dataset in data]

    ## Train model
    ## -----------

    # First layer: layer1
    if 'alpha_min' in layer1:
        alpha_min = layer1.pop('alpha_min')
        layer1['log_alpha_min'] = numpy.log(alpha_min)
    if 'alpha_max' in layer1:
        alpha_max = layer1.pop('alpha_max')
        layer1['log_alpha_max'] = numpy.log(alpha_max)

    print 'Training layer1 (%s)' % layer1['name']
    rbm1 = exp.create_rbm(conf, layer1, data, model=layer1['name'])
    print 'processing data through layer1'
    data = [utils.sharedX(rbm1.function()(dataset.get_value(borrow=True)),
                          borrow=True)
                for dataset in data]

    # Compute train ALC
    if conf.get('transfer', False):
        print 'Computing train ALC'
        data_train, label_train = utils.filter_labels(data[0], label)
        alc = embed.score(data_train, label_train)
        print '... resulting ALC on train is', alc
        conf['train_alc'] = alc

    # Final layer (alternatively): transductive PCA on valid/test
    print 'Learning transductive PCA on the validation set'
    pca_valid = exp.create_pca(conf, trans_pca_valid, data,
            model=trans_pca_valid['name'])
    print 'processing valid data by said transductive PCA'
    valid_repr = pca_valid.function()(data[1].get_value(borrow=True))

    print 'Learning transductive PCA on the test set'
    pca_test = exp.create_pca(conf, trans_pca_test, data,
            model=trans_pca_test['name'])
    print 'processing valid data by said transductive PCA'
    test_repr = pca_test.function()(data[2].get_value(borrow=True))

    # Save submission
    utils.save_submission(conf, valid_repr, test_repr)

    # Re-sync state
    for dname in ('conf', 'layer1', 'trans_pca_valid', 'trans_pca_test'):
        state[dname].update(locals()[dname])

    return channel.COMPLETE

