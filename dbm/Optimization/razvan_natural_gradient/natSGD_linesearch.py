"""
Notes:
    1) Code depends on the latest version of Theano
        (you need my pull request to fix Rop and the new interface of scan
    2) Dependency on pylearn2 was dropped, though the code was meant to be
        part of pylearn2

This code implements natural gradient where we use a Jacobi preconditioner
when trying to invert the metric

Razvan Pascanu
"""

import numpy
import theano.tensor as TT
import theano
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams
#from pylearn2.optimization import minres
#from pylearn2.optimization import scalar_armijo_search, scalar_search_wolfe2
#from pylearn2.utils import constant as const
import minres
import time
from theano.ifelse import ifelse
from optimize import linesearch

gpu_mode = theano.Mode(linker='vm')
cpu_mode = theano.Mode(linker='cvm').excluding('gpu')

def const(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))


class natSGD_linesearch(object):
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

        eps = numpy.float32(1e-24)
        xdata = theano.shared(data['train_x'],
                              name='xdata')
        ydata = theano.shared(data['train_y'],
                           name='ydata')
        self.xdata = xdata
        self.ydata = ydata
        shared_data = [xdata, ydata]

        self.rng = numpy.random.RandomState(options['seed'])
        n_samples = data['train_x'].shape[0]
        self.grad_batches = n_samples // options['gbs']
        self.metric_batches = n_samples // options['mbs']
        self.eval_batches = n_samples // options['ebs']

        self.verbose = options['verbose']

        # Store eucledian gradients
        self.gs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]
        # Store riemannian gradients
        self.rs1 = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]
        self.rs2 = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]
        # Store jacobi diagonal
        self.js = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
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
            # Compute jacobi
            nw_outs = safe_clone(model.outs, replace=replace)
            final_results = dict(zip(model.params, [None]*n_params))
            for nw_out, out_operator in zip(nw_outs, model.outs_operator):
                if out_operator == 'sigmoid':
                    denom = numpy.float32(options['cbs'])
                    #denom *= nw_out
                    #denom *= (numpy.float32(1) - nw_out)
                elif out_operator == 'softmax':
                    denom = numpy.float32(options['cbs'])
                    denom *= (nw_out + eps)
                else:
                    denom = numpy.float32(options['cbs'])
                factor = TT.sqrt(numpy.float32(1) / denom)
                if out_operator == 'sigmoid':
                    tnwout = TT.nnet.sigmoid(nw_out)
                    factor = TT.sqrt(tnwout * (numpy.float32(1) -
                                               tnwout))*factor
                r = TT.sgn(srng.normal(nw_out.shape, nstreams=128))
                r = r * factor
                loc_params = [x for x in model.params if
                              x in theano.gof.graph.inputs([nw_out])]
                jvs = TT.Lop(nw_out, loc_params, r)
                for lp, lj in zip(loc_params, jvs):
                    if final_results[lp] is None:
                        final_results[lp] = TT.sqr(lj)
                    else:
                        final_results[lp] = final_results[lp] + TT.sqr(lj)
            nw_js = [oj + final_results[p] for oj, p in
                     zip(args[1+n_params:1+2*n_params], model.params)]
            return [args[0] + const(1)] + nw_gs + nw_js

        ig = [TT.unbroadcast(TT.alloc(const(0), 1, *shp),0)
              for shp in model.params_shape]
        ij = [TT.unbroadcast(TT.alloc(const(options['jreg']), 1, *shp),0)
              for shp in model.params_shape]
        idx0 = TT.unbroadcast(const([0]),0)
        n_steps = options['gbs'] // options['cbs']
        rvals, updates = scan(grad_step,
                              states=[idx0] + ig + ij,
                              n_steps=n_steps,
                              mode=gpu_mode,
                              name='grad_loop',
                              profile=options['profile'])

        nw_gs = [x[0] / const(n_steps) for x in rvals[1: 1 + n_params]]
        nw_js = [x[0] for x in rvals[1+n_params:1+2*n_params]]
        updates.update(dict(zip(self.gs + self.js, nw_gs + nw_js)))
        grad_inps = [(x, y[gbdx*options['gbs']:(gbdx+1)*options['gbs']])
                     for x,y in zip(loc_inputs, shared_data)]


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
        #theano.printing.pydotprint(self.compute_eucledian_gradients,
        #        'eucledian_grad', scan_graphs=True)

        self.damping = theano.shared(numpy.float32(options['mreg']))
        # Step 2.1 Compile function for Computing Riemannian gradients
        rbdx = TT.iscalar('riemmanian_batch_idx')
        rbpos = rbdx * options['mbs']
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
                nw_outs = safe_clone(model.gf_outs, replace)
                final_results = dict(zip(model.params, [None] * len(model.params)))
                for nw_out, out_operator in zip(nw_outs, model.gf_outs_operator):
                    loc_params = [x for x in model.params
                                  if x in theano.gof.graph.inputs([nw_out])]
                    loc_args = [x for x, y in zip(args, model.params)
                                if y in theano.gof.graph.inputs([nw_out])]
                    if out_operator == 'softmax':
                        factor = const(options['cbs']) * (nw_out + eps)
                    elif out_operator == 'sigmoid':
                        factor = const(options['cbs'])# * nw_out * (1 - nw_out)
                    else:
                        factor = const(options['cbs'])
                    if out_operator != 'sigmoid':
                        loc_Gvs = TT.Lop(nw_out, loc_params,
                                     TT.Rop(nw_out, loc_params, loc_args) /\
                                     factor)
                    else:
                        tnwout = TT.nnet.sigmoid(nw_out)
                        loc_Gvs = TT.Lop(nw_out, loc_params,
                                         TT.Rop(nw_out, loc_params,
                                                loc_args) *\
                                         tnwout * (1 - tnwout)/ factor)

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
            #_final_Gvs = [x + self.damping * y 
            #        for x,y in zip(final_Gvs, args)]
            return final_Gvs, updates


        print 'Constructing riemannian gradient function'
        norm_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in self.gs))

        rvals = minres.minres(
            compute_Gv,
            [x / norm_grads for x in self.gs],
            #Ms = self.js,
            rtol=options['mrtol'],
            shift= self.damping,
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


        updates.update(dict(zip(self.rs1, nw_rs)))
        grad_inps = [(x, y[rbdx * options['mbs']:
                           (rbdx + 1) * options['mbs']])
                     for x,y in zip(loc_inputs, shared_data)]
        print 'Compiling riemannian gradient function'
        self.compute_riemannian_gradients1 = theano.function(
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

        # Step 2.2 Compile function for Computing Riemannian gradients
        rbpos = rbdx * options['mbs']
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
                nw_outs = safe_clone(model.gc_outs, replace)
                final_results = dict(zip(model.params, [None] * len(model.params)))
                for nw_out, out_operator in zip(nw_outs, model.gc_outs_operator):
                    loc_params = [x for x in model.params
                                  if x in theano.gof.graph.inputs([nw_out])]
                    loc_args = [x for x, y in zip(args, model.params)
                                if y in theano.gof.graph.inputs([nw_out])]
                    if out_operator == 'softmax':
                        factor = const(options['cbs']) * (nw_out + eps)
                    elif out_operator == 'sigmoid':
                        factor = const(options['cbs'])# * nw_out * (1 - nw_out)
                    else:
                        factor = const(options['cbs'])
                    if out_operator != 'sigmoid':
                        loc_Gvs = TT.Lop(nw_out, loc_params,
                                     TT.Rop(nw_out, loc_params, loc_args) /\
                                     factor)
                    else:
                        tnwout = TT.nnet.sigmoid(nw_out)
                        loc_Gvs = TT.Lop(nw_out, loc_params,
                                         TT.Rop(nw_out, loc_params,
                                                loc_args) *\
                                         tnwout * (1 - tnwout)/ factor)

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
            #_final_Gvs = [x + self.damping * y 
            #        for x,y in zip(final_Gvs, args)]
            return final_Gvs, updates


        print 'Constructing riemannian gradient function'
        norm_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in self.gs))

        rvals = minres.minres(
            compute_Gv,
            [x / norm_grads for x in self.gs],
            #Ms = self.js,
            rtol=options['mrtol'],
            shift= self.damping,
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


        updates.update(dict(zip(self.rs2, nw_rs)))
        grad_inps = [(x, y[rbdx * options['mbs']:
                           (rbdx + 1) * options['mbs']])
                     for x,y in zip(loc_inputs, shared_data)]
        print 'Compiling riemannian gradient function'
        self.compute_riemannian_gradients2 = theano.function(
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
        if options['rsch'] == 1:
            self.rs = self.rs1
        else:
            self.rs = self.rs2

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

        final_cost = rvals[1].sum() / const(n_steps)
        grad_inps = [(x, y[ebdx * options['ebs']:
                           (ebdx + 1) * options['ebs']])
                     for x,y in zip(loc_inputs, shared_data)]
        denom = -lr*sum([TT.sum(g*r) for g,r in zip(self.gs, self.rs)])
        self.approx_change = theano.function(
                [lr],
                denom,
                name='approx_change',
                mode=gpu_mode,
                allow_input_downcast = True,
                profile=options['profile'])

        print 'compling evaluation function'
        self.eval_fn = theano.function(
            [ebdx, lr],
            final_cost,
            givens=dict(grad_inps),
            on_unused_input='warn',
            updates = updates,
            name='eval_fn',
            allow_input_downcast=True,
            mode=gpu_mode,
            profile=options['profile'])


        
        def ls_grad_step(_idx, gws):
            idx = TT.cast(_idx, 'int32')
            nw_inps = [x[idx * options['cbs']: 
                (idx + 1) * options['cbs']] for x in loc_inputs]
            replace = dict(zip(model.inputs + model.params,
                               nw_inps + nw_ps))
            nw_cost = safe_clone(model.train_cost, replace=replace)
            nw_gs = TT.grad(nw_cost, lr)
            return _idx + numpy.float32(1), gws + nw_gs

        states = [TT.constant(numpy.float32([0])),
                  TT.constant(numpy.float32([0]))]
        rvals, _ = scan(ls_grad_step,
                        states = states,
                        n_steps = n_steps,
                        name = 'ls_grad_step',
                        profile=options['profile'])

        fgrad = rvals[1][0] / const(n_steps)
        self.grad_lr_fn = theano.function(
            [ebdx, lr],
            fgrad,
            givens = grad_inps,
            name='ls_grad_fn',
            on_unused_input='warn',
            mode=gpu_mode,
            allow_input_downcast=True,
            profile=options['profile'])

        update_dict = dict(zip(model.params, nw_ps))
        self.update_params = theano.function(
            [lr],
            [],
            updates=update_dict,
            name='update_params',
            on_unused_input='warn',
            allow_input_downcast=True,
            mode=mode,
            profile=options['profile'])

        self.options = options
        self.old_cost = numpy.inf
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
                        mode=gpu_mode,
                        profile = options['profile'])
        ferr = rvals[1][0] / const(n_steps)
        self.compute_error = theano.function([ebdx],
                           ferr,
                           givens=dict(grad_inps),
                           name='compute_err',
                           mode=gpu_mode,
                           on_unused_input='warn',
                           profile=options['profile'])

    def find_optimum(self, pos):
        ls_cost = lambda x: self.eval_fn(pos, x)
        ls_grad = lambda x: self.grad_lr_fn(pos, x)

        derphi0 = ls_grad(numpy.float32(0))
        phi0 = ls_cost(numpy.float32(0))
        aopt, score, _ = linesearch.scalar_search_wolfe1(
                                ls_cost,
                                ls_grad,
                                phi0 = phi0,
                                derphi0 = derphi0)
        if aopt is None:
            print 'Switching to python wolfe2'
            aopt, score, _, _ = linesearch.scalar_search_wolfe2(
                                ls_cost,
                                ls_grad,
                                phi0 = phi0,
                                derphi0=derphi0)

        use_armijo = False
        try:
            use_armijo = (score > self.old_score*2. or
                          aopt is None or
                          score is None or
                          numpy.isnan(score) or
                          numpy.isinf(score) or
                          numpy.isinf(aopt) or
                          numpy.isinf(aopt))
        except:
            use_armijo = True

        if use_armijo:
            print 'Trying armijo linesearch'
            alpha0 = 1.
            tmp_cost = ls_cost(alpha0)
            _step = 0
            while not numpy.isfinite(tmp_cost) \
                    and tmp_cost > phi0 and _step < 100:
                alpha0 = alpha0/2.
                _step += 1
                tmp_cost = ls_cost(alpha0)
            aopt, score = linesearch.scalar_search_armijo(
                        ls_cost, phi0=phi0, derphi0=derphi0,
                        alpha0 = alpha0)
        if aopt is None or score is None:
            score = numpy.nan
        else:
            self.old_score = score
        return score, aopt


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
        g_st = time.time()
        self.compute_eucledian_gradients(self.permg[self.posg])
        g_ed = time.time()
        r_st = time.time()
        rvals1 = self.compute_riemannian_gradients1(self.permr[self.posr])
        rvals2 = self.compute_riemannian_gradients2(self.permr[self.posr])
        r_ed = time.time()
        e_st = time.time()
        f_0 = self.eval_fn(self.perme[self.pose], 0)
        #f_1em9 = self.eval_fn(self.perme[self.pose], 1e-0)
        #f_1em8 = self.eval_fn(self.perme[self.pose], 1e-8)
        #f_1em7 = self.eval_fn(self.perme[self.pose], 1e-7)
        #f_1em6 = self.eval_fn(self.perme[self.pose], 1e-6)
        #f_1em5 = self.eval_fn(self.perme[self.pose], 1e-5)
        f_1em4 = self.eval_fn(self.perme[self.pose], 1e-4)
        f_1em3 = self.eval_fn(self.perme[self.pose], 1e-3)
        f_1em2 = self.eval_fn(self.perme[self.pose], 1e-2)
        f_5em2 = self.eval_fn(self.perme[self.pose], .05)
        f_1em1 = self.eval_fn(self.perme[self.pose], 1e-1)
        f_5em1 = self.eval_fn(self.perme[self.pose], .5)
        f_1em0 = self.eval_fn(self.perme[self.pose], 1.)
        #a_1em9 = self.approx_change(1e-9)
        #a_1em8 = self.approx_change(1e-8)
        #a_1em7 = self.approx_change(1e-7)
        #a_1em6 = self.approx_change(1e-6)
        #a_1em5 = self.approx_change(1e-5)
        a_1em4 = self.approx_change(1e-4)
        a_1em3 = self.approx_change(1e-3)
        a_1em2 = self.approx_change(1e-2)
        a_5em2 = self.approx_change(.05)
        a_1em1 = self.approx_change(1e-1)
        a_5em1 = self.approx_change(.5)
        a_1em0 = self.approx_change(1.)
        #rho_1em9 = (f_1em9 - f_0)/a_1em9
        #rho_1em8 = (f_1em8 - f_0)/a_1em8
        #rho_1em7 = (f_1em7 - f_0)/a_1em7
        #rho_1em6 = (f_1em6 - f_0)/a_1em6
        #rho_1em5 = (f_1em5 - f_0)/a_1em5
        rho_1em4 = (f_1em4 - f_0)/a_1em4
        rho_1em3 = (f_1em3 - f_0)/a_1em3
        rho_1em2 = (f_1em2 - f_0)/a_1em2
        rho_5em2 = (f_5em2 - f_0)/a_5em2
        rho_1em1 = (f_1em1 - f_0)/a_1em1
        rho_5em1 = (f_5em1 - f_0)/a_5em1
        rho_1em0 = (f_1em0 - f_0)/a_1em0
        #cost, step = self.find_optimum(self.perme[self.pose])
        #if rho_1em9 < .5 or not numpy.isfinite(rho_1em9):
        #    step = 1e-11
        #    cost = self.eval_fn(self.perme[self.pose], 1e-11)
        #elif rho_1em8 < .7 or not numpy.isfinite(rho_1em8):
        #    step = 1e-9
        #    cost = f_1em9
        #if rho_1em7 < .7 or not numpy.isfinite(rho_1em7):
        #    step = 1e-8
        #    cost = f_1em8
        #elif rho_1em6 < .7 or not numpy.isfinite(rho_1em6):
        #    step = 1e-7
        #    cost = f_1em7
        #if rho_1em5 < .7 or not numpy.isfinite(rho_1em5):
        #    step = 1e-6
        #    cost = f_1em6
        #if rho_1em4 < .7 or not numpy.isfinite(rho_1em4):
        #    step = 1e-5
        #    cost = f_1em5
        #if rho_1em3 < .7 or not numpy.isfinite(rho_1em3):
        #    step = 1e-4
        #    cost = f_1em4
        #elif rho_1em2 < .7 or not numpy.isfinite(rho_1em2):
        #    step = 1e-3
        #    cost = f_1em3
        #elif rho_5em2 < .7 or not numpy.isfinite(rho_5em2):
        #    step = .01
        #    cost = f_1em2
        #elif rho_1em1 < .7 or not numpy.isfinite(rho_1em1):
        #    step = .05
        #    cost = f_5em2
        #elif rho_5em1 < .7 or not numpy.isfinite(rho_5em1):
        #    step = .1
        #    cost = f_1em1
        #elif rho_1em0 < .7 or not numpy.isfinite(rho_1em0):
        #    step = .5
        #    cost = f_5em1
        #else:
        #    step = 1.
        #    cost = f_1em0
        step = 1.
        cost = f_1em0
        damping = self.damping.get_value()
        if rho_1em0 < .25:
            self.damping.set_value(numpy.float32(
                self.damping.get_value()*3./2.))
        elif rho_1em0 > .75:
            self.damping.set_value(numpy.float32(
                self.damping.get_value()*2./3.))
        if rho_1em0 > .24:
            self.update_params(step)
        self.old_cost = cost
        e_ed = time.time()
        st = time.time()
        error = self.compute_error(self.perme[self.pose])
        comp_error_time = time.time() - st

        if self.verbose > 1:
            print '----------'
            print 'Minres -- [Gauss Newton]: %s' % minres.msgs[rvals1[0]], \
                        '# iters %04d' % rvals1[1], \
                        'relative error residuals %10.8f' % rvals1[2], \
                        'Anorm', rvals1[4], 'Acond', rvals1[5], \
                        'damping', damping
            print 'Interval:', rvals1[4] - rvals1[4] / rvals1[5]
            print 'Minres -- [Gradient Covariance]: %s' % minres.msgs[rvals2[0]], \
                        '# iters %04d' % rvals2[1], \
                        'relative error residuals %10.8f' % rvals2[2], \
                        'Anorm', rvals2[4], 'Acond', rvals2[5], \
                        'damping', damping
            print 'Interval:', rvals2[4] - rvals2[4] / rvals2[5]
            print '--------------'
        rvals = rvals1

        if self.verbose > 0:
            msg = ('.. iter %4d score %8.5f, error %8.5f step_size %12.9f '
                    #'rho_1em9 %12.9f '
                    #'rho_1em8 %12.9f '
                    #'rho_1em7 %12.9f '
                    #'rho_1em6 %12.9f '
                    #'rho_1em5 %12.9f '
                    'damping %12.9f '
                   'rho_1em4 %12.9f '
                   'rho_1em3 %12.9f '
                   'rho_1em2 %12.9f '
                   'rho_5em2 %12.9f '
                   'rho_1em1 %12.9f '
                   'rho_5em1 %12.9f '
                   'rho_1em0 %12.9f '
                   'ord0_norm %6.3f '
                   'time [grad] %s,'
                   '[riemann grad] %s,'
                   '[updates param] %s,'
                   '%2d(%2d/%2d) [bsg] %2d(%2d/%2d) [bsr] %2d(%2d/%2d)[bse]')
            print msg % (
                self.k,
                cost,
                error,
                step,
                #rho_1em9,
                #rho_1em8,
                #rho_1em7,
                #rho_1em6,
                #rho_1em5,
                damping,
                rho_1em4,
                rho_1em3,
                rho_1em2,
                rho_5em2,
                rho_1em1,
                rho_5em1,
                rho_1em0,
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
            'damping' : damping,
            'step' : step,
            #'rho1em9' : rho_1em9,
            #'rho1em8' : rho_1em8,
            #'rho1em7' : rho_1em7,
            #'rho1em6' : rho_1em6,
            #'rho1em5' : rho_1em5,
            'rho1em4' : rho_1em4,
            'rho1em3' : rho_1em3,
            'rho1em2' : rho_1em2,
            'rho5em2' : rho_5em2,
            'rho1em1' : rho_1em1,
            'rho5em1' : rho_5em1,
            'rho1em0' : rho_1em0,
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

