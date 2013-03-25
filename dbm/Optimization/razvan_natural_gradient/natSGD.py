"""
Notes:
    1) Code depends on the latest version of Theano
        (you need my pull request to fix Rop and the new interface of scan
    2) Dependency on pylearn2 was dropped, though the code is meant to be
        made part of pylearn2

This code implements natural gradient

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

gpu_mode = theano.Mode(linker='vm')
cpu_mode = theano.Mode(linker='cvm').excluding('gpu')

def const(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))


class natSGD(object):
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
                `mbs` -> int
                    Number of samples over which to compute the metric
                `ebs` -> int
                    Number of samples over which to evaluate the training
                    error
                `mreg` -> float
                    Regularization added to the metric
                `mrtol` -> float
                    Relative tolerance for inverting the metric
                `miters` -> int
                    Number of iterations
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
        n_params = len(model.params)
        self.data = data

        if options['device'] != 'gpu':
            xdata = theano.shared(data['train_x'][:options['gbs']],
                                  name='xdata')
            ydata = TT._shared(data['train_y'][:options['gbs']],
                               name='ydata')
            self.xdata = xdata
            self.ydata = ydata
            shared_data = [xdata, ydata]
        else:
            self.cpu_shared_data = []
            xdata = theano.shared(data['train_x'], name='xdata')
            ydata = TT._shared(data['train_y'], name='ydata')
            self.xdata = xdata
            self.ydata = ydata
            shared_data = [xdata, ydata]

        self.rng = numpy.random.RandomState(options['seed'])
        n_samples = data['train_x'].shape[0]
        self.grad_batches = n_samples // options['gbs']
        self.metric_batches = n_samples // options['mbs']
        self.eval_batches = n_samples // options['ebs']

        self.verbose = options['verbose']
        if options['device'] != 'gpu':
            # Store eucledian gradients
            self.gs = [TT._shared(numpy.zeros(shp, dtype=theano.config.floatX))
                       for shp in model.params_shape]
            # Store riemannian gradients
            self.rs = [TT._shared(numpy.zeros(shp, dtype=theano.config.floatX))
                       for shp in model.params_shape]
        else:
            # Store eucledian gradients
            self.gs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                       for shp in model.params_shape]
            # Store riemannian gradients
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

        # inputs
        gbdx = TT.iscalar('grad_batch_idx')
        print 'Constructing grad function'
        srng = RandomStreams(numpy.random.randint(1e5))
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

        # updates
        updates.update(dict(zip(self.gs, nw_gs)))
        # givens
        if options['device'] == 'gpu':
            grad_inps = [(x, y[gbdx*options['gbs']:(gbdx+1)*options['gbs']])
                     for x,y in zip(loc_inputs, shared_data)]
        else:
            grad_inps = zip(loc_inputs, shared_data)

        print 'Compiling grad function'
        self.compute_eucledian_gradients = theano.function(
            [gbdx],
            [],
            updates=updates,
            givens=dict(grad_inps),
            name='compute_eucledian_gradients',
            mode=gpu_mode,
            on_unused_input='warn',
            profile=options['profile'])

        # Step 2. Compile function for Computing Riemannian gradients
        rbdx = TT.iscalar('riemmanian_batch_idx')
        rbpos = rbdx * options['mbs']

        if options['device'] == 'gpu':
            mode=gpu_mode
            def compute_Gv(*args):
                idx0 = const([0])
                ep = [TT.alloc(const(0), 1, *shp)
                      for shp in model.params_shape]

                def Gv_step(*gv_args):
                    idx = TT.cast(gv_args[0], 'int32')
                    nw_inps = [x[idx * options['cbs']: \
                                 (idx + 1) * options['cbs']] for x in
                               loc_inputs]
                    replace = dict(zip(model.inputs, nw_inps))
                    nw_outs = safe_clone(model.outs, replace)
                    final_results = dict(zip(model.params, [None] * len(model.params)))
                    for nw_out, out_operator in zip(nw_outs, model.outs_operator):
                        loc_params = [x for x in model.params
                                      if x in theano.gof.graph.inputs([nw_out])]
                        loc_args = [x for x, y in zip(args, model.params)
                                    if y in theano.gof.graph.inputs([nw_out])]
                        if out_operator == 'softmax':
                            factor = const(options['cbs']) * nw_out
                        elif out_operator == 'sigmoid':
                            factor = const(options['cbs']) * nw_out * (1 - nw_out)
                        else:
                            factor = const(options['cbs'])

                        loc_Gvs = TT.Lop(nw_out, loc_params,
                                         TT.Rop(nw_out, loc_params, loc_args) /\
                                         factor)

                        for lp, lgv in zip(loc_params, loc_Gvs):
                            if final_results[lp] is None:
                                final_results[lp] = lgv
                            else:
                                final_results[lp] += lgv

                    Gvs = [ogv + final_results[param]
                           for (ogv, param) in zip(gv_args[1:], model.params)]
                    return [gv_args[0] + const(1)] + Gvs

                    nw_cost, nw_preactiv_out = safe_clone([model.train_cost,
                                                           model.preactiv_out],
                                                          replace)
                    nw_gvs = TT.Lop(nw_preactiv_out, model.params,
                                  TT.Rop(TT.grad(nw_cost, nw_preactiv_out),
                                         model.params, args))

                    Gvs = [ogv + ngv
                           for (ogv, ngv) in zip(gv_args[1:], nw_gvs)]
                    return [gv_args[0] + const(1)] + Gvs
                states = [idx0] + ep
                n_steps = options['mbs'] // options['cbs']
                rvals, updates = scan(Gv_step,
                                      states=states,
                                      n_steps=n_steps,
                                      mode=theano.Mode(linker='cvm'),
                                      name='Gv_step',
                                      profile=options['profile'])

                final_Gvs = [x[0] / const(n_steps) for x in rvals[1:]]
                return final_Gvs, updates
        else:
            mode = cpu_mode
            def compute_Gv(*args):
                cgv = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX),
                                     name ='cgv%d'%idx)
                           for idx, shp in enumerate(model.params_shape)]
                print_mem('allocated mem for cgv')
                idx0 = const([0])
                ep = [TT.alloc(const(0), 1, *shp)
                      for shp in model.params_shape]

                def Gv_step(*gv_args):
                    idx = TT.cast(gv_args[0], 'int32')
                    nw_inps = [x[idx * options['cbs']: \
                                 (idx + 1) * options['cbs']] for x in
                               loc_inputs]
                    replace = dict(zip(model.inputs, nw_inps))
                    nw_outs = safe_clone(model.outs, replace)
                    final_results = dict(zip(model.params, [None] * len(model.params)))
                    for nw_out, out_operator in zip(nw_outs, model.outs_operator):
                        loc_params = [x for x in model.params
                                      if x in theano.gof.graph.inputs([nw_out])]
                        loc_args = [x for x, y in zip(cgv, model.params)
                                    if y in theano.gof.graph.inputs([nw_out])]
                        if out_operator == 'softmax':
                            factor = const(options['cbs']) * nw_out
                        elif out_operator == 'sigmoid':
                            factor = const(options['cbs']) * nw_out * (1 - nw_out)
                        else:
                            factor = const(options['cbs'])

                        loc_Gvs = TT.Lop(nw_out, loc_params,
                                         TT.Rop(nw_out, loc_params, loc_args) /\
                                         factor)

                        for lp, lgv in zip(loc_params, loc_Gvs):
                            if final_results[lp] is None:
                                final_results[lp] = lgv
                            else:
                                final_results[lp] += lgv

                    Gvs = [ogv + final_results[param]
                           for (ogv, param) in zip(gv_args[1:], model.params)]
                    return [gv_args[0] + const(1)] + Gvs
                states = [idx0] + ep
                n_steps = options['mbs'] // options['cbs']
                rvals, updates = scan(Gv_step,
                                      states=states,
                                      n_steps=n_steps,
                                      mode=gpu_mode,
                                      name='Gv_step',
                                      profile=options['profile'])
                final_Gvs = [TT.as_tensor_variable(x[0]) / const(n_steps) for x in rvals[1:]]
                grad_inps = zip(loc_inputs, shared_data)
                loc_fn = theano.function([],
                                         final_Gvs,
                                         updates = updates,
                                         givens = dict(grad_inps),
                                         on_unused_input='warn',
                                         mode=gpu_mode,
                                         name='loc_fn',
                                         profile = options['profile'])
                fake_op = FakeGPUShell(cgv, loc_fn, len(cgv))

                return fake_op(*args), {}



        print 'Constructing riemannian gradient function'
        norm_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in self.gs))
        rvals = minres.minres(
            compute_Gv,
            [x / norm_grads for x in self.gs],
            rtol=options['mrtol'],
            shift= -options['mreg'],
            maxit=options['miters'],
            mode=mode,
            profile=options['profile'])
        nw_rs = [x * norm_grads for x in rvals[0]]
        flag = rvals[1]
        niters = rvals[2]
        rel_residual = rvals[3]
        rel_Aresidual = rvals[4]
        Anorm = rvals[5]
        Acond = rvals[6]
        xnorm = rvals[7]
        Axnorm = rvals[8]
        updates = rvals[9]

        norm_ord0 = TT.max(abs(nw_rs[0]))
        for r in nw_rs[1:]:
            norm_ord0 = TT.maximum(norm_ord0,
                                   TT.max(abs(r)))


        updates.update(dict(zip(self.rs, nw_rs)))
        grad_inps = [(x, y[rbdx * options['mbs']:
                           (rbdx + 1) * options['mbs']])
                     for x,y in zip(loc_inputs[:1], shared_data[:1])]
        print 'Compiling riemannian gradient function'
        self.compute_riemannian_gradients = theano.function(
            [rbdx],
            [flag,
             niters,
             rel_residual,
             rel_Aresidual,
             Anorm,
             Acond,
             xnorm,
             Axnorm,
             norm_grads,
             norm_ord0],
            updates=updates,
            givens=dict(grad_inps),
            name='compute_riemannian_gradients',
            on_unused_input='warn',
            mode=mode,
            profile=options['profile'])

        # Step 3. Compile function for evaluating cost and updating
        # parameters
        print 'constructing evaluation function'
        lr = TT.scalar('lr')
        self.lr = numpy.float32(options['lr'])
        ebdx = TT.iscalar('eval_batch_idx')
        nw_ps = [p - lr * r for p, r in zip(model.params, self.rs)]

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
                              mode=gpu_mode,
                              profile=options['profile'])

        final_cost = rvals[1] / const(n_steps)
        if options['device'] == 'gpu':
            grad_inps = [(x, y[ebdx * options['ebs']:
                           (ebdx + 1) * options['ebs']])
                     for x,y in zip(loc_inputs, shared_data)]
        else:
            grad_inps = zip(loc_inputs, shared_data)

        print 'compling evaluation function'
        self.eval_fn = theano.function(
            [ebdx, lr],
            final_cost,
            givens=dict(grad_inps),
            on_unused_input='warn',
            updates = updates,
            name='eval_fn',
            mode=gpu_mode,
            profile=options['profile'])

        update_dict = dict(zip(model.params, nw_ps))
        if options['device'] != 'gpu':
            update_dict.update(dict(zip(model.cparams, nw_ps)))
        self.update_params = theano.function(
            [lr],
            [],
            updates=update_dict,
            name='update_params',
            on_unused_input='warn',
            mode=mode,
            profile=options['profile'])
        self.options = options
        self.old_cost = 1e6
        self.device = options['device']
        n_steps = options['ebs'] // options['cbs']
        def ls_error(_idx, acc):
            idx = TT.cast(_idx, 'int32')
            nw_inps = [x[idx * options['cbs']: \
                         (idx + 1) * options['cbs']] for x in loc_inputs]
            replace = dict(zip(model.inputs, nw_inps))
            nw_cost = TT.cast(safe_clone(
                model.err, replace=replace), 'float32')
            return [_idx + const(1), acc + nw_cost]

        states = [TT.constant(numpy.float32([0])),
                  TT.constant(numpy.float32([0]))]
        rvals, _ = scan(ls_error,
                        states = states,
                        n_steps = n_steps,
                        name='ls_err_step',
                        mode=cpu_mode,
                        profile = options['profile'])
        ferr = rvals[1][0] / const(n_steps)
        self.compute_error = theano.function([ebdx],
                           ferr,
                           givens=dict(grad_inps),
                           name='compute_err',
                           mode=gpu_mode,
                           on_unused_input='warn',
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
        mbs = self.options['mbs']
        gbs = self.options['gbs']
        ebs = self.options['ebs']
        if self.device == 'gpu':
            g_st = time.time()
            self.compute_eucledian_gradients(self.permg[self.posg])
            g_ed = time.time()
            r_st = time.time()
            rvals = self.compute_riemannian_gradients(self.permr[self.posr])
            r_ed = time.time()
            e_st = time.time()
            cost = self.eval_fn(self.perme[self.pose], self.lr)
            while (self.lr > 1e-6 and
                   (cost > self.old_cost * 1.5 or
                    numpy.isnan(cost) or
                    numpy.isinf(cost))):
                self.lr = self.lr/2.
                cost = self.eval_fn(self.perme[self.pose], self.lr)
            if self.lr < 1e-6:
                raise Exception('Learning rate too small !')
            self.old_cost = cost
            self.update_params(self.lr)
            e_ed = time.time()
            st = time.time()
            error = self.compute_error(self.perme[self.pose])
            comp_error_time = time.time() - st
        else:
            g_st = time.time()
            self.xdata.set_value(self.data['train_x'][self.permg[self.posg]*gbs:
                                                      (self.permg[self.posg]+1)*gbs])
            self.ydata.set_value(self.data['train_y'][self.permg[self.posg]*gbs:
                                                      (self.permg[self.posg]+1)*gbs])
            self.compute_eucledian_gradients(self.permg[self.posg])
            g_ed = time.time()
            r_st = time.time()
            self.xdata.set_value(self.data['train_x'][self.permr[self.posr]*mbs:
                                                      (self.permr[self.posr]+1)*mbs])
            self.ydata.set_value(self.data['train_y'][self.permr[self.posr]*mbs:
                                                      (self.permr[self.posr]+1)*mbs])
            rvals = self.compute_riemannian_gradients(self.permr[self.posr])
            r_ed = time.time()
            e_st = time.time()
            self.xdata.set_value(self.data['train_x'][self.perme[self.pose]*ebs:
                                                      (self.perme[self.pose]+1)*ebs])
            self.ydata.set_value(self.data['train_y'][self.perme[self.pose]*ebs:
                                                      (self.perme[self.pose]+1)*ebs])
            cost = self.eval_fn(self.perme[self.pose], self.lr)
            while (self.lr > 1e-6 and
                   (cost > self.old_cost * 1.5 or
                    numpy.isnan(cost) or
                    numpy.isinf(cost))):
                self.lr = self.lr/2.
                cost = self.eval_fn(self.perme[self.pose], self.lr)
            if self.lr < 1e-6:
                raise Exception('Learning rate too small !')
            self.old_cost = cost
            self.update_params(self.lr)
            e_ed = time.time()
            st = time.time()
            error = self.compute_error(self.perme[self.pose])
            comp_error_time = time.time() - st


        if self.verbose > 1:
            print 'Minres: %s' % minres.msgs[rvals[0]], \
                        '# iters %04d' % rvals[1], \
                        'relative error residuals %10.8f' % rvals[2], \
                        'Anorm', rvals[4], 'Acond', rvals[5]
        if self.verbose > 0:
            msg = ('.. iter %4d score %8.5f, error %8.5f step_size %12.9f '
                   'ord0_norm %6.3f '
                   'time [grad] %s,'
                   '[riemann grad] %s,'
                   '[updates param] %s,'
                   '%2d(%2d/%2d) [bsg] %2d(%2d/%2d) [bsr] %2d(%2d/%2d)[bse]')
            print msg % (
                self.k,
                cost,
                error,
                self.lr,
                rvals[-1],
                print_time(g_ed - g_st),
                print_time(r_ed - r_st),
                print_time(e_ed - e_st),
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
            'error': error,
            'time_err' : comp_error_time,
            'time_grads': g_ed - g_st,
            'time_metric': r_ed - r_st,
            'time_eval': e_ed - e_st,
            'minres_flag': rvals[0],
            'minres_iters': rvals[1],
            'minres_relres': rvals[2],
            'minres_Anorm': rvals[4],
            'minres_Acond': rvals[5],
            'grad_norm': rvals[-1],
            'beta': numpy.float32(0),
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

class FakeGPUShell(theano.gof.Op):
    def __init__(self, args, fn, n_params):
        self.args = args
        self.fn = fn
        self.n_params = n_params

    def __hash__(self):
        # Diff ?
        return hash(type(self))

    def __eq__(self, other):
        # Diff ?
        return type(self) == type(other)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, *args):
        return theano.gof.Apply(self, args, [x.type() for x in args[:self.n_params]])

    def perform(self, node, inputs, outputs):
        for vb, dt in zip(self.args, inputs[:self.n_params]):
            vb.set_value(dt)
        nw_vals =  self.fn(*inputs[self.n_params:])
        for vb, dt in zip(outputs, nw_vals):
            vb[0] = dt

