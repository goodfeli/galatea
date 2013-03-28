"""
Description:
    main.py contains the logic for training a given model with a given
    optimization technique.

    Its purporse is to ensure that all algorithms (for all models) are
    treated the same, and the resuts are stored in a similar fashion,
    in order to make comparison between them very easy.

Notes:
    this code is meant to run via jobman/jobdispatch (method main)

Instructions:
    WRITE ME

Razvan Pascanu
"""

# Generic imports
import numpy
import theano.tensor as TT
import theano
from theano.ifelse import ifelse
import cPickle
import gzip
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import minres
import time

# optimization algorithms
from SGD import SGD
from natSGD_jacobi import natSGD_jacobi
from natSGD_linesearch import natSGD_linesearch
from natNCG import natNCG
from krylov_lbfgs import KrylovDescent
# models
from mlp import mlp
from daa import daa

class MainLoop(object):
    def __init__(self, options, channel):
        # Step 0. Load data
        print 'Loading data'
        data = numpy.load(options['data'])
        self.options = options
        self.channel = channel
        # Step 1. Construct Model
        print 'Constructing Model'
        if options['model'] == 'mlp':
            model = mlp(options, channel, data)
        elif options['model'] == 'daa':
            model = daa(options, channel, data)
        self.model = model
        print 'Constructing algo'
        # Step 2. Construct optimization technique
        if options['algo'] == 'natSGD_basic':
            algo =  natSGD(options, channel, data, model)
        elif options['algo'] == 'natSGD_jacobi':
            algo =  natSGD_jacobi(options, channel, data, model)
        elif options['algo'] == 'natSGD_ls':
            algo = natSGD_linesearch(options, channel, data, model)
        elif options['algo'] == 'natNCG':
            algo = natNCG(options, channel, data, model)
        elif options['algo'] == 'krylov':
            algo = KrylovDescent(options, channel, data, model)
        elif options['algo'] == 'hf':
            raise NotImplemented
        elif options['algo'] == 'hf_jacobi':
            raise NotImplemented
        elif options['algo'] == 'sgd':
            algo = SGD(options, channel, data, model)
        self.algo = algo
        self.options['validscore'] = 1e20
        self.train_timing = numpy.zeros((options['loopIters'],
                                    27), dtype='float32')
        self.valid_timing = numpy.zeros((options['loopIters'],
                                    2), dtype='float32')
        if self.channel is not None:
            self.channel.save()
        self.start_time = time.time()
        self.batch_start_time = time.time()

    def validate(self):
        cost = self.model.valid_eval_func()
        print ('** validation score %6.3f computed in %s'
               ', best score is %6.3f') % (
                   cost,
                   print_time(time.time() - self.batch_start_time),
                   self.options['validscore'])
        self.batch_start_time = time.time()
        if self.options['validscore'] > cost:
            self.options['validscore'] = float(cost)
            self.options['validtime'] = time.time() - self.start_time
            self.test()
            self.save()
            if self.channel is not None:
                self.channel.save()
        print_mem('validate')


    def test(self):
        cost = self.model.test_eval_func()
        print '>>> Test score', cost
        self.options['testscore'] = float(cost)
        if self.channel is not None:
            self.channel.save()

    def save(self):
        numpy.savez('timing.npz',
                    train=self.train_timing,
                    valid=self.valid_timing,
                    k=self.k)
        self.model.save()


    def main(self):
        print_mem('start')
        self.options['gotNaN'] = 0
        self.start_time = time.time()
        self.batch_start_time = time.time()
        self.k = 0
        while self.k < self.options['loopIters']:
            st = time.time()
            rvals = self.algo()
            #print 'One step took', print_time(time.time() - st)
            self.train_timing[self.k, 0] = rvals['time_grads']
            self.train_timing[self.k, 1] = rvals['time_metric']
            self.train_timing[self.k, 2] = rvals['time_eval']
            self.train_timing[self.k, 3] = rvals['score']
            self.train_timing[self.k, 4] = rvals['minres_iters']
            self.train_timing[self.k, 5] = rvals['minres_relres']
            self.train_timing[self.k, 6] = rvals['minres_Anorm']
            self.train_timing[self.k, 7] = rvals['minres_Acond']
            self.train_timing[self.k, 8] = rvals['grad_norm']
            self.train_timing[self.k, 9] = rvals['beta']
            self.train_timing[self.k, 10] = rvals['lambda']
            self.train_timing[self.k, 11] = rvals['error']
            self.train_timing[self.k, 12] = rvals['time_err']
            self.train_timing[self.k, 13] = rvals['damping']
            #self.train_timing[self.k, 14] = rvals['rho1em9']
            #self.train_timing[self.k, 15] = rvals['rho1em8']
            #self.train_timing[self.k, 16] = rvals['rho1em7']
            #self.train_timing[self.k, 17] = rvals['rho1em6']
            #self.train_timing[self.k, 18] = rvals['rho1em5']
            self.train_timing[self.k, 19] = rvals['rho1em4']
            self.train_timing[self.k, 20] = rvals['rho1em3']
            self.train_timing[self.k, 21] = rvals['rho1em2']
            self.train_timing[self.k, 22] = rvals['rho5em2']
            self.train_timing[self.k, 23] = rvals['rho1em1']
            self.train_timing[self.k, 24] = rvals['rho5em1']
            self.train_timing[self.k, 25] = rvals['rho1em0']
            self.train_timing[self.k, 26] = rvals['step']
            if numpy.isinf(rvals['score']) or numpy.isnan(rvals['score']):
                self.options['gotNaN'] = 1
                self.save()
                raise Exception('Got NaN while training')
            self.k += 1
            if 'checkFreq' in self.options and \
               self.k % self.options['checkFreq'] == 0:
                self.validate()

        self.validate()
        print 'BEST SCORE'
        print 'Validation', self.options['validscore']
        print 'Validation time', print_time(self.options['validtime'])
        print 'TEST', self.options['testscore']


def main(options, channel):
    main_loop = MainLoop(options, channel)
    main_loop.main()

################## UTILITY FUNCTIONS ###############################
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self,data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

import sys
sys.stdout=Unbuffered(sys.stdout)

def print_mem(context=None):
    if theano.sandbox.cuda.cuda_enabled:
        rvals = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        # Avaliable memory in Mb
        available = float(rvals[0]) / 1024. / 1024.
        # Total memory in Mb
        total = float(rvals[1]) / 1024. / 1024.
        if context == None:
            print ('Used %.3f Mb Free  %.3f Mb, total %.3f Mb' %
                   (total - available, available, total))
        else:
            info = str(context)
            print (('GPU status : Used %.3f Mb Free %.3f Mb,'
                    'total %.3f Mb [context %s]') %
                    (total - available, available, total, info))

def print_time(secs):
    if secs < 120.:
        return '%6.3f sec' % secs
    elif secs <= 60*60:
        return '%6.3f min' % (secs / 60.)
    else:
        return '%6.3f h  ' % (secs / 3600.)


if __name__ == '__main__':
    # Set up some defaults !!
    options= {}
    options['profile'] = 1
    #options['data'] = '/home/pascanur/data/cifar10.npz'
    options['data'] = '/home/pascanur/data/mnist.npz'
    options['verbose'] = 3
    options['device'] = 'gpu'
    # batch for computing gradient
    options['gbs'] = 50000
    # batch for computing the metric
    options['mbs'] = 25000
    # batch for evaluating the model
    options['ebs'] = 25000
    # number of samples to consider at any time
    options['cbs'] = 500
    #options['model'] = 'mlp'
    options['model'] = 'mlp'
    #options['algo'] = 'sgd'
    #options['algo'] = 'krylov'
    #options['algo'] = 'natNCG'
    options['algo'] = 'natSGD_ls'
    #options['algo'] = 'natSGD_jacobi'
    #options['algo'] = 'natSGD_basic'
    options['type'] = 'Gf' #'gradCov' # Gf
    options['adaptivedamp'] = 0.
    options['loopIters'] = 400
    options['gotNaN'] = 0
    options['seed'] = 123
    #options['hids'] = '[1000, 500, 250, 30]'
    options['hids'] = '[1000,1000, 1000]'
    # weight initialization formula .. not very useful to change it right now
    options['init'] = 'Xavier' #'small'
    # error cost for deep autoencoder (note Dumi and I think Martens used cross entropy for MNIST)
    options['daacost'] = 'cross'
    options['mrtol'] =0
    options['mreg'] = 32.
    options['jreg'] = .2
    options['rsch'] = 1
    options['resetFreq'] = 40
    options['miters'] = numpy.int32(60)
    options['lr'] = .1
    options['lsIters'] = 80
    options['checkFreq'] = 5
    options['krylovDim'] = 15
    options['lbfgsIters'] = 30
    main(options, None)
