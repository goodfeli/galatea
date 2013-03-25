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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import minres
import time
strlinker = 'vm'
gpu_mode = theano.Mode(linker='vm')
cpu_mode = theano.Mode(linker='cvm').excluding('gpu')

class mlp(object):
    def __init__(self, options, channel, data):
        self.rng = numpy.random.RandomState(options['seed'])
        self.srng = RandomStreams(self.rng.randint(1e5))
        self.nin = data['train_x'].shape[1]
        self.options = options
        if isinstance(options['hids'], list):
            self.hids = options['hids']
        else:
            self.hids = eval(str(options['hids']))
        self.nout = numpy.int32(numpy.max(data['train_y']) + 1)
        def gen_mat(nin, nout, name):
            # NOTE : assumes sigmoid
            self.rng = numpy.random.RandomState(123)
            if options['init'] == 'small':
                lim = numpy.sqrt(1./nin)
                vals = self.rng.uniform(size=(nin, nout), low=-lim,
                        high=lim).astype('float32')
            else:
                lim = numpy.sqrt(6./(nin + nout))
                print 'Lim used to generate random numbers', lim
                vals = self.rng.uniform(size=(nin, nout), low=-lim,
                               high=lim).astype('float32') * 4.
                try:
                    print 'Rank (',nin, ',', nout, '):', \
                            numpy.linalg.matrix_rank(vals)
                except:
                    pass
            var = theano.shared(vals, name=name)
            print_mem(name)
            return var

        def gen_vec(n, name):
            self.rng = numpy.random.RandomState(123)
            vals = self.rng.uniform(size=(n,), low=-.0005,
                                 high=.0005).astype('float32')*0.
            var = theano.shared(vals, name=name)
            print_mem(name)
            return var


        ##### PARAMS
        all_hids = [self.nin] + self.hids + [self.nout]
        activs = [TT.nnet.sigmoid] * len(self.hids) + [softmax]
        #activs = [TT.tanh] * len(self.hids) + [softmax]

        self.params = []
        self.cpu_params = []
        self.params_shape = []
        for idx, (in_dim, out_dim) in\
                    enumerate(zip(all_hids[:-1], all_hids[1:])):
            gpu_W = gen_mat(in_dim, out_dim, name='W%d'%idx)
            gpu_b = gen_vec(out_dim, name='b%d'%idx)
            self.params += [gpu_W, gpu_b]
            self.params_shape.append((in_dim, out_dim))
            self.params_shape.append((out_dim,))

        self.x = TT.matrix('X')
        self.y = TT.ivector('y')
        self.inputs = [self.x, self.y]
        hid = self.x
        for idx, activ in zip(range(len(self.params)//2), activs):
            W = self.params[idx * 2]
            b = self.params[idx * 2 + 1]
            preactiv = TT.dot(hid, W) + b
            hid = activ(preactiv)
        self.preactiv_out = preactiv
        batch_train_cost =-TT.log(hid)[
                    TT.constant(numpy.asarray(range(options['cbs'])).astype('int32')),
                    self.y]

        if options['type'] == 'gradCov':
            self.outs = [batch_train_cost]
            self.outs_operator = ['linear']
        elif options['type'] == 'leroux':
            self.outs = [batch_train_cost - batch_train_cost.mean()]
            self.outs_operator = ['linear']
        else:
            self.outs = [hid]
            self.outs_operator = ['softmax']

        self.gf_outs = [hid]
        self.gf_outs_operator = ['softmax']
        self.gc_outs = [batch_train_cost]
        self.gc_outs_operator = ['linear']

        self.train_cost = TT.mean(batch_train_cost)
        pred = TT.argmax(hid, axis=1)
        self.err = TT.mean(TT.neq(pred, self.y))
        self.valid_xdata = theano.shared(data['valid_x'],
                                      name='valid_xdata',
                                      borrow=True)
        self.test_xdata = theano.shared(data['test_x'],
                                     name='test_xdata',
                                     borrow=True)
        mode = gpu_mode
        self.valid_ydata = TT.cast(
            theano.shared(data['valid_y'], name='valid_ydata',
                       borrow=True), 'int32')
        self.test_ydata = TT.cast(
            theano.shared(data['test_y'], name='test_xdata',
                       borrow=True), 'int32')


        givens = {}
        givens[self.x] = self.valid_xdata
        givens[self.y] = self.valid_ydata

        self.valid_eval_func = theano.function([],
                                               self.err,
                                               givens=givens,
                                               name='valid_eval_fn',
                                               profile=options['profile'],
                                               mode=mode)

        givens[self.x] = self.test_xdata
        givens[self.y] = self.test_ydata
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

