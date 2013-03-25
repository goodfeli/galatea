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

class rnn(object):
    def _ _init__(self, options, channel, data):

        self.rng = numpy.random.RandomState(options['seed'])
        self.srng = RandomStreams(self.rng.randint(1e5))
        self.nin = data['train_x'].shape[2]
        self.in_shape = (options['cbs'], self.nin)
        self.options = options
        if isinstance(options['hid'], str):
            self.nhid = eval(options['nhid'])
        else:
            self.nhid = options['nhid']
        self.nout = data['train_y'].shape[2]
        def gen_mat(nin, nout, name, device='cpu', scale=.01):
            # NOTE : assumes tanh
            self.rng = numpy.random.RandomState(123)
            vals = self.rng.uniform(size=(nin, nout), low=-scale,
                               high=scale).astype('float32')
            if device=='gpu':
                var = theano.shared(vals, name=name)
                print_mem(name)
            else:
                var = TT._shared(vals, name=name)
            return var

        def gen_vec(n, name, device='cpu'):
            self.rng = numpy.random.RandomState(123)
            vals = self.rng.uniform(size=(n,), low=-.0005,
                                 high=.0005).astype('float32')
            if device=='gpu':
                var = theano.shared(vals, name=name)
                print_mem(name)
            else:
                var = TT._shared(vals, name=name)
            return var
        ##### PARAMS
        Wxx = gen_mat(self.nhid, self.nhid, name='Wxx', device='gpu')
        Wux = gen_mat(self.nin, self.nhid, name='Wux', device='gpu')
        Wxy = gen_mat(self.nhid, self.nout, name='Wxy', device='gpu')
        Wuy = gen_mat(self.nin, self.nout, name='Wuy', device='gpu')
        bx = gen_vec(self.nhid, name='bx', device='gpu')
        self.h0 = gen_mat(options['cbs'], self.nhid, name='h0',
                          device='gpu', scale=0)
        self.params = [Wxx, Wux, Wxy, Wuy, bx, self.h0]
        self.params_shape = [(self.nhid, self.nhid),
                             (self.nin, self.nhid),
                             (self.nhid, self.nout),
                             (self.nin, self.nout),
                             (self.nhid),
                             (options['cbs'], self.nhid) ]

        self.cparams =[]
        self.x = TT.tensor3('X')
        self.y = TT.tensor3('y')

        self.inputs = [self.x, self.y]

        def step(u_t, h_tm1, Wxx, Wux, Wxy, Wuy):
            h_t = TT.tanh(TT.dot(u_t, Wux) + TT.dot(h_tm1, Wxx))
            y_t = TT.dot(h_t, Wxy) + TT.dot(u_t, Wuy)
            return h_t, y_t
        _hid0 = TT.alloc(numpy.float32(0),
                        numpy.int32(options['seqlen']+1),
                        numpy.int32(options['cbs']),
                        numpy.int32(self.nhid))
        hid0 = TT.set_subtensor(hid0[0], self.h0)

        [H,Y], _ = scan(step,
                        self.x,
                        [hid0, None],
                        [Wxx, Wux, Wxy, Wuy],
                        n_sptes = options['seqlen'])
        # TODO : compute 3D cost ...

        if options['device'] == 'cpu/gpu':
            self.cpu_params = [
                TT._shared(x.get_value(), name=x.name) for x in self.params]
            self.err = safe_clone(self.err,
                                  updates=zip(self.params, self.cpu_params))
            self.valid_xdata = TT._shared(data['valid_x'],
                                          name='valid_xdata',
                                          borrow=True)
            self.test_xdata = TT._shared(data['test_x'],
                                         name='test_xdata',
                                         borrow=True)
            mode = cpu_mode
        else:
            self.valid_xdata = theano.shared(data['valid_x'],
                                          name='valid_xdata',
                                          borrow=True)
            self.test_xdata = theano.shared(data['test_x'],
                                         name='test_xdata',
                                         borrow=True)
            mode = gpu_mode
        self.valid_ydata = TT.cast(
            TT._shared(data['valid_y'], name='valid_ydata',
                       borrow=True), 'int32')
        self.test_ydata = TT.cast(
            TT._shared(data['test_y'], name='test_xdata',
                       borrow=True), 'int32')

        givens = {}
        givens[self.x] = self.valid_xdata
        givens[self.y] = self.valid_ydata

        self.valid_eval_func = theano.function([],
                                               ferr,
                                               givens=givens,
                                               name='valid_eval_fn',
                                               profile=options['profile'],
                                               mode=mode)

        givens[self.x] = self.test_xdata
        givens[self.y] = self.test_ydata
        self.test_eval_func = theano.function([],
                                    ferr,
                                    givens=givens,
                                    name='test_fn',
                                    profile=options['profile'],
                                    mode=mode)


    def save(self):
        if 'allgpu' in self.options and self.options['allgpu']:
            vals=dict([(x.name, x.get_value()) for x in self.params])
        else:
            vals=dict([(x.name, x.get_value()) for x in self.cparams])
        numpy.savez('mlp', vals)


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

