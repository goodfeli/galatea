"""
MLP class

Notes:
    It only works for classification tasks !!

Razvan Pascanu
"""

import numpy
import theano.tensor as TT
import theano
from theano.ifelse import ifelse
import cPickle
import gzip
from theano.tensor.shared_randomstreams import RandomStreams
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import minres
import time
strlinker = 'vm'
gpu_mode = theano.Mode(linker='vm')
cpu_mode = theano.Mode(linker='cvm').excluding('gpu')

class daa(object):
    def __init__(self, options, channel, data):

        self.rng = numpy.random.RandomState(options['seed'])
        self.srng = RandomStreams(self.rng.randint(1e5))
        self.nin = data['train_x'].shape[1]
        self.in_shape = (options['cbs'], self.nin)
        self.options = options
        if isinstance(options['hids'], list):
            self.hids = options['hids']
        else:
            self.hids = eval(str(options['hids']))
        nhids = len(self.hids)
        self.nout = numpy.int32(numpy.max(data['train_y']) + 1)
        def gen_mat(nin, nout, name, scale = 1.):
            # NOTE : assumes sigmoid
            self.rng = numpy.random.RandomState(123)
            if options['init'] == 'small':
                lim = numpy.sqrt(1./nin)
                vals = self.rng.uniform(size=(nin, nout), low=-lim,
                        high=lim).astype('float32')
            elif options['init'] == 'sparse':
                vals = numpy.zeros((nin, nout))
                for dx in xrange(nout):
                    perm = self.rng.permutation(nin)
                    vals[perm[:25],dx] = self.rng.normal(size=(25,), scale=1, loc=0)
                vals = vals.astype('float32')
            else:
                lim = numpy.sqrt(6./(nin + nout))
                vals = self.rng.uniform(size=(nin, nout), low=-lim,
                               high=lim).astype('float32') *4.
            vals = vals * scale
            var = theano.shared(vals, name=name)
            print_mem(name)
            return var

        def gen_vec(n, name):
            self.rng = numpy.random.RandomState(123)
            vals = self.rng.uniform(size=(n,), low=-.0005,
                                 high=.0005).astype('float32')
            vals = numpy.zeros_like(vals)
            var = theano.shared(vals, name=name)
            print_mem(name)
            return var
        ##### PARAMS
        layer_sizes = [self.nin] + self.hids
        linear = lambda x:x
        activs = [TT.nnet.sigmoid] * (nhids-1) + [linear]
        #rec_activs = [linear] + [TT.nnet.sigmoid] * (nhids-1) #+ [linear]
        rec_activs = [TT.nnet.sigmoid] * (nhids-1) + [linear]
        self.params = []
        self.cparams = []
        self.params_shape = []
        Ws =[]
        bs =[]
        rbs = []
        rWs = []
        for idx, (in_dim, out_dim) in\
                    enumerate(zip(layer_sizes[:-1],
                                  layer_sizes[1:])):
            if idx == len(layer_sizes) - 2:
                scale = .25
            else:
                scale = 1.
            gW = gen_mat(in_dim, out_dim, name='W%d'%idx, scale=scale)
            gb = gen_vec(out_dim, name='b%d'%idx)
            grb = gen_vec(in_dim, name='rb%d'%idx)
            if idx == len(layer_sizes)-2:
                scale = .25
            else:
                scale = 1.
            grW = gen_mat(out_dim, in_dim, name='rW%d'%idx, scale=scale)
            self.params += [gW, gb, grW, grb]
            Ws += [gW]
            bs += [gb]
            rbs += [grb]
            rWs += [grW]
            self.params_shape.append((in_dim, out_dim))
            self.params_shape.append((out_dim,))
            self.params_shape.append((out_dim, in_dim))
            self.params_shape.append((in_dim,))

        self.x = TT.matrix('X')
        self.inputs = [self.x]

        hid =self.x
        print 'Forward pass'
        for W,b, activ in zip(Ws, bs, activs):
            preactiv = TT.dot(hid, W) + b
            print 'activation', activ
            hid = activ(preactiv)

        rec = hid
        print 'Backwards pass'
        for rW,rb, ractiv in zip(rWs, rbs, rec_activs)[::-1]:
            rec_preactiv = TT.dot(rec, rW) + rb
            print 'activation', ractiv
            rec = ractiv(rec_preactiv)

        print 'done'
        noisy_hid = hid
        noisy_rec = rec
        if options['daacost'] == 'mse':
            batch_rec_cost = ((noisy_rec - self.x)**2)
        elif options['daacost'] == 'cross':
            batch_rec_cost = - (self.x * TT.log(rec) +
                                      (1-self.x)*TT.log(1-rec))
        if options['type'] == 'gradCov':
            self.outs = [batch_rec_cost]
            self.outs_operator= ['linear']
        elif options['type'] == 'leroux':
            self.outs = [batch_rec_cost - batch_rec_cost.mean()]
            self.outs_operator = ['linear']
        else:
            if options['daacost'] == 'mse':
                self.outs = [noisy_rec]
                self.outs_operator = ['linear']
            else:
                self.outs = [rec_preactiv]
                self.outs_operator = ['sigmoid']
        self.preactiv_out = noisy_rec


        self.gf_outs = [rec_preactiv]
        self.gf_outs_operator = ['sigmoid']
        self.gc_outs = [batch_rec_cost]
        self.gc_outs_operator = ['linear']
        self.gc2_outs = [batch_rec_cost.sum(axis=1)]
        self.gc2_outs_operator = ['linear']


        self.train_cost = batch_rec_cost.sum(axis=1).mean()
        if 'l2norm' in options:
        
            self.train_cost += options['l2norm']* sum(TT.sum(x**2) for x in Ws) + \
                    options['l2norm']* sum(TT.sum(x**2) for x in rWs)
        
        self.err = ((rec - self.x)**2).sum(axis=1).mean()
        self.valid_xdata = theano.shared(data['valid_x'],
                                      name='valid_xdata',
                                      borrow=True)
        self.test_xdata = theano.shared(data['test_x'],
                                     name='test_xdata',
                                     borrow=True)
        mode = gpu_mode

        givens = {}
        givens[self.x] = self.valid_xdata

        self.valid_eval_func = theano.function([],
                                               self.err,
                                               givens=givens,
                                               name='valid_eval_fn',
                                               profile=options['profile'],
                                               mode=mode)

        givens[self.x] = self.test_xdata
        self.test_eval_func = theano.function([],
                                    self.err,
                                    givens=givens,
                                    name='test_fn',
                                    profile=options['profile'],
                                    mode=mode)


    def save(self):
        vals=dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez('model', vals)

def softmax(x):
    e = TT.exp(x)
    return e / TT.sum(e, axis=1).dimshuffle(0,'x')

def print_time(secs):
    if secs < 120.:
        return '%6.3f sec' % secs
    elif secs <= 60*60:
        return '%6.3f min' % (secs / 60.)
    else:
        return '%6.3f h  ' % (secs / 3600.)

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

def safe_clone(cost, replace):
    params = replace.keys()
    nw_vals = replace.values()
    dummy_params = [x.type() for x in params]
    dummy_cost = theano.clone(cost,
                              replace=dict(zip(params, dummy_params)))
    return theano.clone(dummy_cost,
                        replace=dict(zip(dummy_params, nw_vals)))

