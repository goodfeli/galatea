import numpy
import os
import sys
import theano
import theano.tensor as tensor
from pylearn.gd.sgd import sgd_updates

from framework.pca import PCA
from framework import utils
from framework.cost import MeanSquaredError
from framework.utils import BatchIterator
from framework.autoencoder import Autoencoder,ContractingAutoencoder
from framework.optimizer import SGDOptimizer
from framework.utils import safe_update

def pp(s,e):
    return theano.printing.Print(s)(e)

class ProgressiveAutoencoder(Autoencoder):
    def __init__(self,*args,**kwargs):
        super(ProgressiveAutoencoder,self).__init__(*args,**kwargs)
#        Autoencoder.__init__(self,*args,**kwargs)
        self.full_weights=self.weights
        self.full_w_prime=self.w_prime
        self.full_visbias=self.visbias

    def reconstruct(self, inputs, mask):
        self.mask=mask
        self.weights=self.full_weights[:self.mask,:]
        self.w_prime=self.full_w_prime[:,:self.mask]
        self.visbias=self.full_visbias[:self.mask]
        return Autoencoder.reconstruct(self,inputs)

    def reset_masking(self):
        self.weights=self.full_weights
        self.w_prime=self.full_w_prime
        self.visbias=self.full_visbias

class ProgressiveCAE(ProgressiveAutoencoder,ContractingAutoencoder):
    def __init__(self,*args,**kwargs):
        super(ProgressiveCAE,self).__init__(*args,**kwargs)
#        ProgressiveAutoencoder.__init__(self,*args,**kwargs)
#        ContractingAutoencoder.__init__(self,*args,**kwargs)

def mask(x,m):
    """
    returns x with the m first entries not masked to 0.
    """
    return x*map(int,numpy.arange(x.shape[-1])<m)

if __name__ == "__main__":

    conf = {
        'dataset':'avicenna',
        'corruption_level': 0.1,
        'nhid': 50,
        'nvis': 75,
        'anneal_start': 100,
        'base_lr': 0.01,
        'tied_weights': True,
        'act_enc': 'tanh',
        'act_dec': None,
        'irange': 0.001,
        'pca_size':75,
       #'unmasking':None,
        #'unmasking':'linear',
       #'unmasking':'sigmoidal',
       'unmasking':'sqrt',
       #'unmasking':'exp',
    }
    data = utils.load_data(conf)
    train,valid,test=[s.get_value() for s in data]

    # Allocate a PCA transformation block.
    pca_model_file = conf['dataset']+'.pca_%i.pkl'%conf['pca_size']
    if os.path.isfile(pca_model_file):
        print '... loading precomputed PCA transform'
        pca = PCA.load(pca_model_file)
    else:
        print '... computing PCA transform'
        pca = PCA(conf['pca_size'],whiten=True)
        pca.train(train)
        pca.save(pca_model_file)

    # A symbolic input representing your minibatch.
    minibatch = tensor.matrix()

    # Create a model
    model = ProgressiveCAE(conf['nvis'], conf['nhid'],
                                conf['act_enc'], conf['act_dec'])
    # Allocate an optimizer, which tells us how to update our model.
    mask = tensor.iscalar()
    if isinstance(model,ProgressiveAutoencoder):
        cost = MeanSquaredError(model)(model.reconstruct(pca(minibatch)[:,:mask],mask),
                                                            minibatch[:,:mask])
    else:
        cost = MeanSquaredError(model)(model.reconstruct(pca(minibatch)),
                                                            minibatch)
    if isinstance(model,ContractingAutoencoder):
        cost+=model.contraction_penalty(minibatch[:,:mask])
    trainer = SGDOptimizer( model,
                            conf['base_lr'],
                            conf['anneal_start'],)
    updates = trainer.cost_updates(cost)

    # Finally, build a Theano function out of all this.
    train_fn = theano.function([minibatch,mask], cost, updates=updates)

    # Suppose we want minibatches of size 10
    batchsize = 25

    print '%s training with %s unmasking'%(model.__class__.__name__,conf['unmasking'])
    # Here's a manual training loop. I hope to have some classes that
    # automate this a litle bit.
    iterator=BatchIterator(data, [1,0,0], batchsize)
    n_epochs=20
    start=conf['pca_size']/10
    import math
    if conf['unmasking']==None:
        unmasking=lambda p:conf['pca_size']
    if conf['unmasking']=='linear':
        unmasking=lambda p:start+p*conf['pca_size']
    elif conf['unmasking']=='sigmoidal':
        sigmoid=lambda x:1./(1+math.e**-x)
        unmasking=lambda p:start+75*sigmoid((p-.5)*10.)
    elif conf['unmasking']=='sqrt':
        unmasking=lambda p:conf['pca_size']*(min(1.,p+.05))**.5
    elif conf['unmasking']=='exp':
        unmasking=lambda p:conf['pca_size']*(math.e**(p*2))/math.e**2

    for epoch in xrange(n_epochs):
        mb_err=[]
        p=epoch/float(n_epochs)
        m=unmasking(p)
        print 'm %i '%m,
        m=min(m,conf['pca_size'])
        for mb in iterator:
            mb_err.append(train_fn(mb,int(m)))
            #mb_err.append(train_fn(mb))
        print "epoch %d: %f" % (epoch, numpy.mean(mb_err))
    if isinstance(model,ProgressiveAutoencoder):
        model.reset_masking()

    print 'training done'
    # reconsruction error on the valid dataset:
    # 50 units 20 epochs pca75 no whitening
    # no unmasking: 33.7424647734
    # linear: 10.2438032069
    # sigmoidal: 12.0239010324
    # sqrt: 13.0689095123
    # exp: 2515.29030416
    rec_err = theano.function([minibatch,mask], cost)

    print "mean reconstruction error on the valid subset:"
    print numpy.mean(rec_err(valid,conf['pca_size']))
