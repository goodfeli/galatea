# Standard library imports
import math
import sys
import time
import os
from tempfile import TemporaryFile
import zipfile

from jobman import DD
import numpy
import theano
from theano import tensor

try:
    import framework
except ImportError:
    print >> sys.stderr, \
        "Framework couldn't be imported. Make sure you have the " \
        "repository root on your PYTHONPATH (or as your current " \
        "working directory)"
    sys.exit(1)

from auc import embed
import framework.cost
import framework.corruption

import framework.utils as utils
from framework.pca import PCA, CovEigPCA, SVDPCA
from framework.utils import BatchIterator
from framework.base import StackedBlocks
from framework.autoencoder import Autoencoder, DenoisingAutoencoder, ContractingAutoencoder
from framework.rbm import GaussianBinaryRBM, PersistentCDSampler
from framework.optimizer import SGDOptimizer

#import framework.experiments.progressiveunmasking.subclasses as feps
from subclasses import ProgressiveAutoencoder, ProgressiveCAE
#hack
framework.autoencoder.ProgressiveAutoencoder = ProgressiveAutoencoder
framework.autoencoder.ProgressiveCAE = ProgressiveCAE


def create_pca(conf, layer, data, model=None):
    """
    Simple wrapper to either load a PCA or train it and save its parameters
    """
    savedir = utils.getboth(layer, conf, 'savedir')
    clsname = layer.get('pca_class', 'SVDPCA')

    if True:
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
            print 'loading layer:', clsname
            try:
                return PCA.load(filename)
            except Exception, e:
                print 'Warning: error while loading %s:' % clsname, e.args[0]
                print 'Switching back to training mode.'

    # Train the model
    print 'training layer:', clsname
    MyPCA = framework.pca.get(clsname)
    pca = MyPCA.fromdict(layer)

    proba = utils.getboth(layer, conf, 'proba')
    blended = utils.blend(data, proba)
    pca.train(blended.get_value(borrow=True))

    if True:
        pca.save(filename)
    return pca


def create_ae(conf, layer, data, model=None):
    savedir = utils.getboth(layer, conf, 'savedir')
    clsname = layer['autoenc_class']

    if False:
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
            print 'loading layer:', clsname
            try:
                return Autoencoder.load(filename)
            except Exception, e:
                print 'Warning: error while loading %s:' % clsname, e.args[0]
                print 'Switching back to training mode.'

    # Set visible units size
    layer['nvis'] = utils.get_constant(data[0].shape[1]).item()

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Retrieve the corruptor object (if needed)
   #name = layer.get('corruption_class', 'DummyCorruptor')
#    MyCorruptor = framework.corruption.get(name)
#    corruptor = MyCorruptor(layer.get('corruption_level', 0))

    # Allocate an denoising or contracting autoencoder
    cae = ContractingAutoencoder.fromdict(layer)#, corruptor=corruptor)
    pcae = ProgressiveCAE.fromdict(layer)#, corruptor=corruptor)

    #pcae
    # Allocate an optimizer, which tells us how to update our model.
    MyCost = framework.cost.get(layer['cost_class'])
    mask = tensor.iscalar()
    varcost_pcae = MyCost(pcae)(minibatch[:,:mask], pcae.reconstruct(minibatch[:,:mask], mask))
    alpha = layer.get('contracting_penalty', 0.1)
    varcost_pcae= tensor.mean(varcost_pcae+ alpha * pcae.contraction_penalty(minibatch[:,:mask]))
    input_size = 75
    start = input_size / 10.
    if layer['unmasking'] == 'linear':
        _unmasking = lambda p:start + p * (input_size * 2)
    elif layer['unmasking'] == 'sigmoidal':
        sigmoid = lambda x:1. / (1. + math.e ** -x)
        _unmasking = lambda p:start + input_size * sigmoid((p-.5) * 10.)
    elif layer['unmasking'] == 'sqrt':
        _unmasking = lambda p:input_size * (min(1., p + .1)) ** .5
    elif layer['unmasking'] == 'exp':
        _unmasking = lambda p:input_size * (math.e ** (p * 2)) / math.e ** 2
    else:
        _unmasking = lambda p:input_size
    #validity ensuring decorator
    unmasking = lambda p:int(min(max(0, _unmasking(p)), input_size))
    #
    #classic cae
    varcost_cae = MyCost(cae)(minibatch, cae.reconstruct(minibatch))
    alpha = layer.get('contracting_penalty', 0.1)
    varcost_cae = tensor.mean(varcost_cae + alpha * cae.contraction_penalty(minibatch))

    trainer = SGDOptimizer(pcae, layer['base_lr'], layer['anneal_start'])
    updates_pcae = trainer.cost_updates(varcost_pcae)

    trainer = SGDOptimizer(cae, layer['base_lr'], layer['anneal_start'])
    updates_cae = trainer.cost_updates(varcost_cae)

    # Finally, build a Theano function out of all this.
    train_fn_pcae = theano.function([minibatch, mask], varcost_pcae,
                               updates=updates_pcae,
                               name='train_fn_pcae')

    train_fn_cae = theano.function([minibatch], varcost_cae,
                               updates=updates_cae,
                               name='train_fn_cae')

    # Here's a manual training loop.
    print 'training both'
    print 'with %s unmasking for the pcae' % layer['unmasking']
    start_time = time.clock()
    proba = utils.getboth(layer, conf, 'proba')
    iterator = BatchIterator(data, proba, layer['batch_size'])
    saving_counter = 0
    saving_rate = utils.getboth(layer, conf, 'saving_rate', 0)
    for epoch in xrange(layer['epochs']):
        err_pcae = []
        err_cae = []
        msk = unmasking(epoch / float(layer['epochs']))
        batch_time = time.clock()
        for minibatch_data in iterator:
            err_pcae.append(train_fn_pcae(minibatch_data, int(msk)))
            err_cae.append(train_fn_cae(minibatch_data))
        # Print training time + cost
        train_time = time.clock() - batch_time
        err_pcae=numpy.mean(err_pcae)
        err_cae=numpy.mean(err_cae)
        print 'epoch %d, msk %i, time spent %f, cost CAE:%f cost PCAE:%f' % \
            (epoch, msk, train_time,err_cae,err_pcae),
        print 'diff (CAE-PCAE): ',err_cae-err_pcae

        # Saving intermediate models
        if saving_rate != 0:
            saving_counter += 1
            if saving_counter % saving_rate == 0:
                cae.save(os.path.join(savedir,
                        layer['name'] + '-epoch-%02d.pkl' % epoch))
                pcae.save(os.path.join(savedir,
                        layer['name'] + '-epoch-%02d.pkl' % epoch))
    pcae.reset_masking()

    end_time = time.clock()
    layer['training_time'] = (end_time - start_time) / 60.
    print 'training ended after %f min' % layer['training_time']

    # Compute denoising error for valid and train datasets.
    #cae
    error_fn = theano.function([minibatch], varcost_cae, name='error_fn')
    layer['error_valid'] = error_fn(data[1].get_value(borrow=True)).item()
    layer['error_test'] = error_fn(data[2].get_value(borrow=True)).item()
    print '== CAE =='
    print 'final error with valid is', layer['error_valid']
    print 'final error with test  is', layer['error_test']

    # pcae
    error_fn = theano.function([minibatch, mask], varcost_pcae, name='error_fn_pcae')
    layer['error_valid'] = error_fn(data[1].get_value(), input_size).item()
    layer['error_test'] = error_fn(data[2].get_value(), input_size).item()
    print '== PCAE =='
    print 'final error with valid is', layer['error_valid']
    print 'final error with test  is', layer['error_test']

    # Save model parameters
    if False:
        cae.save(filename)
        pcae.save(filename)
        print 'final model has been saved as %s' % filename

    # Return the autoencoder object
    return cae,pcae

def state2layer_conf(conf,level):
    newconf={}
    prefix='L'+str(level)+'__'
    for k,v in conf.items():
        if k.startswith(prefix):
            k=k[4:]
            if isinstance(v,tuple):
                v=v[0]
            newconf[k]=v
    return newconf

def run_experiment(state, channel):

    experiment_start = time.clock()

    # Load the dataset
    data = utils.load_data(state)
    if state['transfer']:
        # Data for the ALC proxy
        labels = data[3]
        data = data[:3]

    # First layer : train or load a PCA
    layer1=state2layer_conf(state,1)
    pca1 = create_pca(state, layer1, data, model=layer1['name'])
    data = [utils.sharedX(pca1.function()(s.get_value(borrow=True)), borrow=True)
                                                                for s in data]

    # Second layer : train or load a DAE or CAE
    layer2=state2layer_conf(state,2)
    cae,pcae = create_ae(state, layer2, data)#, model=layer2['name'])
    data_cae = [utils.sharedX(cae.function()(s.get_value(borrow=True)), borrow=False)
                                                                for s in data]
    data_pcae = [utils.sharedX(pcae.function()(s.get_value(borrow=True)), borrow=False)
                                                                for s in data]

    # Third layer : train or load a PCA
    #cae
    layer3_cae=state2layer_conf(state,3)
    pca2_cae = create_pca(state, layer3_cae, data_cae)#, model=layer3['name'])
    data_cae = [utils.sharedX(pca2_cae.function()(s.get_value(borrow=False)), borrow=True)
                                                                for s in data_cae]
    #pcae
    layer3_pcae=state2layer_conf(state,3)
    pca2_pcae = create_pca(state, layer3_pcae, data_pcae)#, model=layer3['name'])
    data_pcae = [utils.sharedX(pca2_pcae.function()(s.get_value(borrow=False)), borrow=True)
                                                                for s in data_pcae]

    # Compute the ALC for example with labels
    if state['transfer']:
        print 'resulting ALC on train is'
        tr_data, tr_labels = utils.filter_labels(data_cae[0], labels)
        alc = embed.score(tr_data, tr_labels)
        state.train_alc_cae = alc
        print 'CAE:', alc
        tr_data, tr_labels = utils.filter_labels(data_pcae[0], labels)
        alc = embed.score(tr_data, tr_labels)
        state.train_alc_pcae = alc
        print 'PCAE:', alc

    # Stack both layers and create submission file
    blocks_cae = StackedBlocks([pca1,
                          cae,
                          pca2_cae,
                          ])
    blocks_pcae = StackedBlocks([pca1,
                          pcae,
                          pca2_pcae,
                          ])
    for stack in [blocks_cae]:#,blocks_pcae]:
        #local
        utils.create_submission(state, stack.function())
        #works on condor
        #submit(data[1], data[2],state.savedir+'/'+layer2['name']+ '_SUBMISSION_.zip')

    state.time_spent = time.clock() - experiment_start

    return channel.COMPLETE

def get_default_hyperparams():
    # Experiment specific arguments
    state=DD()
    state.dataset='avicenna'
    state.expname='pcae'
    state.transfer=True
    state.normalize=True
    state.normalize_on_the_fly=False
    state.randomize_valid=True
    state.randomize_test=True
    state.saving_rate=0
    state.savedir='./'

    #layer 1 PCA-75 whiten
    state.L1__name='avicenna'
    state.L1__num_components=75
    state.L1__min_variance= 0
    state.L1__whiten=True
    state.L1__pca_class='SVDPCA'
    # Training properties
    state.L1__proba=[1, 0, 0]

    # Second layer may vary
    state.L2__name='PCAE'
    state.L2__nhid=250
    state.L2__tied_weights=True
    state.L2__act_enc='sigmoid'
    state.L2__act_dec=None
    state.L2__irange=0.001
    state.L2__cost_class='SquaredError'
    state.L2__autoenc_class='ContractingAutoencoder'
    #'L2__autoenc_class': 'ProgressiveAutoencoder'
    state.L2__autoenc_class='ProgressiveCAE'
    #'L2__autoenc_class': 'DenoisingAutoencoder'
    state.L2__unmasking='sqrt' # linear, sigmoid, exp, sqrt, None
    state.L2__corruption_class='BinomialCorruptor'
    state.L2__corruption_level=0.3
    state.L2__contracting_penalty= 0.1
    state.L2__base_lr=1e-6
    state.L2__anneal_start=100
    state.L2__batch_size=1
    state.L2__epochs=50
    state.L2__proba=[1, 0, 0]

    # layer 3 PCA-3 whiten
    state.L3__name='3st-PCA'
    state.L3__pca_class='SVDPCA'
    state.L3__num_components=7
    state.L3__min_variance=0
    state.L3__whiten=True
    # Training properties
    state.L3__proba=[0, 1, 0]

    return state

def submit(valid, test, filepath, valid_fn="harry_lisa_valid.prepro", test_fn="harry_lisa_final.prepro"):
    valid_file = TemporaryFile()
    test_file = TemporaryFile()

    numpy.savetxt(valid_file, valid, fmt="%.3f")
    numpy.savetxt(test_file, test, fmt="%.3f")

    valid_file.seek(0)
    test_file.seek(0)

    submission = zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED)

    submission.writestr(valid_fn, valid_file.read())
    submission.writestr(test_fn, test_file.read())

    submission.close()
    valid_file.close()
    test_file.close()

if __name__ == "__main__":
    run_experiment(get_default_hyperparams(),DD(COMPLETE=0))
