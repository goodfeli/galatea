"""
Notes:
    1) Code depends on the latest version of Theano
        (you need my pull request to fix Rop and the new interface of scan
    2) Dependency on pylearn2 was dropped, though the code is meant to be
        made part of pylearn2

This code implements SGD

Razvan Pascanu
"""

import numpy
import theano.tensor as TT
import theano
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from pylearn2.optimization import minres
#from pylearn2.optimization import scalar_armijo_search, scalar_search_wolfe2
#from pylearn2.utils import constant as const
import minres
import time
from theano.ifelse import ifelse


def const(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))


class SGD(object):
    def __init__(self, options, channel, data, model):
        """
        Parameters:
            options: Dictionary
            `options` is expected to contain the following keys:
                `cbs` -> int
                    Number of samples to consider at a time when computing
                    some property of the model
                `gbs` -> int
                    Number of samples over which to compute the gradients
                `ebs` -> int
                    Number of samples over which to evaluate the training
                    error
                `seed` -> int
                    Random number generator seed
                `profile` -> bool
                    Flag, if profiling should be on or not
                `verbose` -> int
                    Verbosity level
                `lr` -> float
                    Learning rate
            channel: jobman channel or None
            data: dictionary-like object return by numpy.load containing the
                data
            model : model
        """
        self.model = model
        # push dataset into shared var
        n_params = len(model.params)
        xdata = theano.shared(data['train_x'].astype('float32'),
                              name='xdata')
        # ! This works for 1 of k classification
        ydata = TT.cast(
            theano.shared(data['train_y'].astype('float32'),
                          name='ydata'), 'int32')

        shared_data = [xdata, ydata]
        self.xdata = xdata
        self.ydata = ydata
        # all sorts of indices
        self.rng = numpy.random.RandomState(options['seed'])
        n_samples = data['train_x'].shape[0]
        self.grad_batches = n_samples // options['gbs']
        self.metric_batches = n_samples // options['mbs']
        self.eval_batches = n_samples // options['ebs']

        self.verbose = options['verbose']

        # vars for gradients
        # Store Euclidean gradients
        self.gs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]
        # Store riemannian gradients (H^-1*g)
        self.rs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]
        self.permg = self.rng.permutation(self.grad_batches)
        self.permr = self.rng.permutation(self.metric_batches)
        self.perme = self.rng.permutation(self.eval_batches)
        self.k = 0
        self.posg = 0
        self.posr = 0
        self.pose = 0

        # Step 1. Compile function for computing eucledian gradients
        gbdx = TT.iscalar('grad_batch_idx')
        print 'Constructing grad function'
        loc_inputs = [x.type() for x in model.inputs]
        def grad_step(*args):
            idx = TT.cast(args[0], 'int32')
            nw_inps = [x[idx * options['cbs']: \
                         (idx + 1) * options['cbs']]
                       for x in loc_inputs]
            replace = dict(zip(model.inputs, nw_inps))
            nw_cost = safe_clone(model.train_cost, replace=replace)
            gs = TT.grad(nw_cost, model.params)
            nw_gs = [op + np for op, np in zip(args[1: 1 + n_params], gs)]
            return [args[0] + const(1)] + \
                    nw_gs

        ig = [TT.unbroadcast(TT.alloc(const(0), 1, *shp),0)
              for shp in model.params_shape]
        idx0 = TT.unbroadcast(const([0]),0)
        n_steps = options['gbs'] // options['cbs']
        rvals, updates = scan(grad_step,
                              states=[idx0] + ig,
                              n_steps=n_steps,
                              name='grad_loop',
                              profile=options['profile'])

        nw_gs = [x[0] / const(n_steps) for x in rvals[1: 1 + n_params]]
        updates.update(dict(zip(self.gs, nw_gs)))
        grad_inps = [(x, y[gbdx*options['gbs']:(gbdx+1)*options['gbs']])
                     for x,y in zip(loc_inputs, shared_data)]
        print 'Compiling grad function'
        self.compute_eucledian_gradients = theano.function(
            [gbdx],
            [],
            updates=updates,
            givens=dict(grad_inps),
            on_unused_input='warn',
            name='compute_eucledian_gradients',
            mode=theano.Mode(linker='cvm'),
            profile=options['profile'])

        # Step 3. Compile function for evaluating cost and updating
        # parameters
        print 'constructing evaluation function'
        lr = TT.scalar('lr')
        self.lr = numpy.float32(options['lr'])
        ebdx = TT.iscalar('eval_batch_idx')
        nw_ps = [p - lr * g for p, g in zip(model.params, self.gs)]

        def cost_step(_idx, acc):
            idx = TT.cast(_idx, 'int32')
            nw_inps = [x[idx * options['cbs']: \
                         (idx + 1) * options['cbs']] for x in loc_inputs]
            replace = dict(zip(model.inputs + model.params, nw_inps + nw_ps))
            nw_cost = safe_clone(model.train_cost, replace=replace)
            return [_idx + const(1),
                    acc + nw_cost]

        acc0 = const([0])
        idx0 = const([0])
        n_steps = options['ebs'] // options['cbs']
        rvals, updates = scan(cost_step,
                              states=[idx0, acc0],
                              n_steps=n_steps,
                              name='cost_loop',
                              profile=options['profile'])

        final_cost = rvals[1] / const(n_steps)
        update_vals = dict(zip(model.params, nw_ps))
        #updates.update(dict(zip(model.params, nw_ps)))
        grad_inps = [(x, y[ebdx * options['ebs']:
                           (ebdx + 1) * options['ebs']])
                     for x,y in zip(loc_inputs, shared_data)]

        print 'compling evaluation function'
        self.eval_fn = theano.function(
            [ebdx, lr],
            final_cost,
            givens=dict(grad_inps),
            updates= updates,
            on_unused_input='warn',
            name='eval_fn',
            mode=theano.Mode(linker='cvm'),
            profile=options['profile'])
        self.update_params = theano.function(
            [lr],
            [],
            updates=update_vals,
            on_unused_input='warn',
            #givens=dict(grad_inps),
            name='update_params',
            mode=theano.Mode(linker='cvm'),
            profile=options['profile'])
        self.options = options
        self.old_cost = 1e6

        n_steps = options['ebs'] // options['cbs']
        def ls_error(_idx, acc, acc_train_cost):
            idx = TT.cast(_idx, 'int32')
            nw_inps = [x[idx * options['cbs']: \
                         (idx + 1) * options['cbs']] for x in loc_inputs]
            replace = dict(zip(model.inputs, nw_inps))
            nw_cost = TT.cast(safe_clone(model.err,
                                         replace=replace),'float32')
            train_cost = TT.cast(safe_clone(model.train_cost,
                                          replace=replace), 'float32')
            return [_idx + const(1),
                    acc + nw_cost,
                    acc_train_cost + train_cost]

        states = [TT.constant(numpy.float32([0])),
                  TT.constant(numpy.float32([0])),
                  TT.constant(numpy.float32([0]))]
        rvals, _ = scan(ls_error,
                        states = states,
                        n_steps = n_steps,
                        name='ls_err_step',
                        mode=theano.Mode(linker='cvm'),
                        profile = options['profile'])

        ferr = rvals[1][0] / const(n_steps)
        ftrain_cost = rvals[2][0] / const(n_steps)

        self.compute_error = theano.function([ebdx],
                           [ferr, ftrain_cost],
                           givens=dict(grad_inps),
                           name='compute_err',
                           on_unused_input='warn',
                           mode=theano.Mode(linker='cvm'),
                           profile=options['profile'])


    def __call__(self):
        """
        returns: dictionary
            the dictionary contains the following entries:
                'cost': float - the cost evaluted after current step
                'time_grads': time wasted to compute gradients
                'time_metric': time wasted to compute the riemannian
                     gradients
                'time_eval': time wasted to evaluate function
                'minres_flag': flag indicating the output of minres
                'minres_iters': number of iteration done by minres
                'minres_relres': relative error of minres
                'minres_Anorm': norm of the metric
                'minres_Acond': condition number of the metric
                'grad_norm': gradient norm
                'beta': beta factor for directions in conjugate gradient
                'lambda: lambda factor
        """
        if self.posg == self.grad_batches:
            self.permg = self.rng.permutation(self.grad_batches)
            self.posg = 0
        if self.posr == self.metric_batches:
            self.permr = self.rng.permutation(self.metric_batches)
            self.posr = 0
        if self.pose == self.eval_batches:
            self.perme = self.rng.permutation(self.eval_batches)
            self.pose = 0
        g_st = time.time()
        self.compute_eucledian_gradients(self.permg[self.posg])
        g_ed = time.time()
        e_st = time.time()
        cost = self.eval_fn(self.perme[self.pose], self.lr)
        while (self.lr > 1e-6 and
               (cost > 3 * self.old_cost or
                numpy.isnan(cost) or
                numpy.isinf(cost))):
            print cost, self.old_cost, self.lr
            self.lr = self.lr/2.
            cost = self.eval_fn(self.perme[self.pose], self.lr)
        if self.lr < 1e-6:
            raise Exception('Learning rate too small !')
        self.old_cost = cost
        self.update_params(self.lr)
        e_ed = time.time()
        st = time.time()
        error, train_cost = self.compute_error(self.perme[self.pose])
        comp_error_time = time.time() - st
        if self.verbose > 0 and self.k % 250 == 0:
            msg = ('.. iter %4d score %8.5f error %8.5f '
                   'train_cost %8.5f '
                   'step %12.9f  '
                   'time [grad] %s '
                   '[updates param] %s '
                   '[error] %s '
                   '%2d(%2d/%2d) [bsg] %2d(%2d/%2d) [bsr] %2d(%2d/%2d)[bse]')
            print msg % (
                self.k,
                cost,
                error,
                train_cost,
                self.lr,
                print_time((g_ed - g_st)*250.),
                print_time((e_ed - e_st)*250.),
                print_time(comp_error_time),
                self.permg[self.posg],
                self.posg + 1,
                self.grad_batches,
                self.permr[self.posr],
                self.posr + 1,
                self.metric_batches,
                self.perme[self.pose],
                self.pose + 1,
                self.eval_batches)

        self.k += 1
        self.pose += 1
        self.posg += 1
        self.posr += 1
        ret = {
            'score': cost,
            'error':error,
            'time_err' : comp_error_time,
            'time_grads': g_ed - g_st,
            'time_metric': 0,
            'time_eval': e_ed - e_st,
            'minres_flag': numpy.nan,
            'minres_iters': numpy.nan,
            'minres_relres': numpy.nan,
            'minres_Anorm': numpy.nan,
            'minres_Acond': numpy.nan,
            'grad_norm': numpy.nan,
            'beta': numpy.float32(0),
            'damping' : numpy.float32(0),
            'rho' : numpy.float32(0),
            'lambda': numpy.float32(0)}
        return ret


def safe_clone(cost, replace):
    params = replace.keys()
    nw_vals = replace.values()
    dummy_params = [x.type() for x in params]
    dummy_cost = theano.clone(cost,
                              replace=dict(zip(params, dummy_params)))
    return theano.clone(dummy_cost,
                        replace=dict(zip(dummy_params, nw_vals)))


def print_time(secs):
    if secs < 120.:
        return '%6.3f sec' % secs
    elif secs <= 60*60:
        return '%6.3f min' % (secs / 60.)
    else:
        return '%6.3f h  ' % (secs / 3600.)

