import numpy
import theano
from theano import tensor
import theano.tensor as TT
from theano.sandbox.scan import scan
import scipy
import scipy.optimize
import time

zero = tensor.constant(numpy.float32(0.))
gpu_mode = theano.Mode(linker='vm')
cpu_mode = theano.Mode(linker='cvm').excluding('gpu')

def safe_clone(cost, replace):
    params = replace.keys()
    nw_vals = replace.values()
    dummy_params = [x.type(name='dummyp') for x in params]
    dummy_cost = theano.clone(cost,
                              replace=dict(zip(params, dummy_params)))
    return theano.clone(dummy_cost,
                        replace=dict(zip(dummy_params, nw_vals)))


def const(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))

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
            print  \
                    (('GPU status : Used %.3f Mb Free %.3f Mb,'
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

def krylov_subspace(compute_Av,
                    bs,
                    old_dir,
                    iters=20,
                    param_shapes=None,
                    profile=0,
                    device='gpu'):
    eps = numpy.float32(1e-20)
    bs = [b / tensor.sqrt((b ** 2).sum()+eps) for b in bs]
    mem_bufs = [tensor.alloc(zero, iters, *param_sh) for
           param_sh in param_shapes]
    mem_bufs = [tensor.set_subtensor(mem[0], b)
                    for mem, b in zip(mem_bufs, bs)]

    def construct_space(*args):
        vs, updates = compute_Av(*args)
        # I need to rescale at every point, otherwise if A is damping, these
        # vs go quickly to 0 and we loose the direction they represent
        norm = TT.sqrt(sum((v**2).sum() for v in vs)) + numpy.float32(1e-20)
        vs = [v / norm for v in vs]
        return vs, updates
    if device == 'gpu':
        mode = gpu_mode
    else:
        mode = cpu_mode
    outs, updates = scan(construct_space,
                   states=mem_bufs,
                   n_steps=iters - 2,
                   name='krylov_space',
                   mode=mode,
                   profile=profile)
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    outs = [tensor.set_subtensor(out[iters - 1], o)
                for out, o in zip(outs, old_dir)]
    outs = [tensor.unbroadcast(tensor.shape_padleft(x), 0)
                for x in outs]
    param_lengths = [numpy.prod(shp) for shp in param_shapes]

    def ortho(idx, *ortho_mats):
        new_ortho_mats = []
        for A, param_length in zip(ortho_mats, param_lengths):
            weight = tensor.dot(A[idx + 1:].reshape(
                (iters - idx - 1, param_length)),
                A[idx].reshape((param_length,)))
            A_reshuffle = ['x'] + list(range(A[idx].ndim))
            W_reshuffle = [0] + ['x'] * A[idx].ndim
            to_remove = weight.dimshuffle(*W_reshuffle) *\
                        A[idx].dimshuffle(*A_reshuffle)
            new_A = tensor.set_subtensor(A[idx + 1:],
                                         A[idx + 1:] - to_remove)
            x_col = new_A[idx + 1]
            x_col = x_col / tensor.sqrt((x_col ** 2).sum()+eps)
            new_A = tensor.set_subtensor(new_A[idx + 1], x_col)
            new_ortho_mats.append(new_A)
        return new_ortho_mats
    rvals, _ = scan(ortho,
                    sequences=tensor.constant(numpy.arange(iters - 1)),
                    states=outs,
                    n_steps=iters - 1,
                    name='ortho',
                    profile=profile,
                    mode=mode)
    if not isinstance(rvals, (list, tuple)):
        rvals = [rvals]
    rvals = [rval[0]*.1 for rval in rvals]
    return rvals, updates


class KrylovDescent(object):
    def __init__(self,
                 options,
                 channel,
                 data,
                 model):
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
                    Number of samples over which to compute the krylov
                    subspace
                `ebs` -> int
                    Number of samples over which to evaluate the training
                    error
                `seed` -> int
                    Random number generator seed
                `profile` -> bool
                    Flag, if profiling should be on or not
                `verbose` -> int
                    Verbosity level
                `lbfgsIters' -> int
                `krylovDim` -> int
            channel: jobman channel or None
            data: dictionary-like object return by numpy.load containing the
                data
            model : model
        """
        n_params = len(model.params)
        self.data = data
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
        rng = numpy.random.RandomState(options['seed'])
        self.rng = rng
        self.options = options
        self.channel = channel
        self.model = model
        n_dimensions = options['krylovDim']
        self.n_dimensions = n_dimensions
        if options['device']=='gpu':
            cfn_subspaces = \
                [theano.shared(numpy.zeros(
                                (n_dimensions,) + shp, dtype='float32'),
                               name='cfn{%s|%d}' % (str(param.name), i))
                 for i, (shp, param) in enumerate(zip(model.params_shape,
                                                      model.params))]
            old_deltas = \
                [theano.shared(numpy.zeros(shp, dtype='float32'),
                               name='delta{%s|%d}' % (str(param.name), i))
                 for i, (shp, param) in
                            enumerate(zip(model.params_shape, model.params))]
            self.gs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]
        else:
            cfn_subspaces = \
                [TT._shared(numpy.zeros(
                                (n_dimensions,) + shp, dtype='float32'),
                               name='cfn{%s|%d}' % (str(param.name), i))
                 for i, (shp, param) in enumerate(zip(model.params_shape,
                                                      model.params))]
            old_deltas = \
                [TT._shared(numpy.zeros(shp, dtype='float32'),
                               name='delta{%s|%d}' % (str(param.name), i))
                 for i, (shp, param) in
                            enumerate(zip(model.params_shape, model.params))]
            self.gs = [TT._shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]
        self.cfn_subspaces = cfn_subspaces
        self.old_deltas = old_deltas

        self.permg = self.rng.permutation(self.grad_batches)
        self.permr = self.rng.permutation(self.metric_batches)
        self.perme = self.rng.permutation(self.eval_batches)
        self.k = 0
        self.posg = 0
        self.posr = 0
        self.pose = 0

        # Step 1. Compile function for computing eucledian gradients
        print 'Constructing grad function'
        loc_inputs = [x.type(name='locx') for x in model.inputs]
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
                              mode=gpu_mode,
                              profile=options['profile'])

        nw_gs = [x[0] / const(n_steps) for x in rvals[1: 1 + n_params]]
        updates.update(dict(zip(self.gs, nw_gs)))
        gdx = TT.iscalar('gdx')
        grad_inps = zip(loc_inputs,
                        [x[gdx*options['gbs']:(gdx+1)*options['gbs']] for x
                         in shared_data])
        print 'Compiling grad function'
        self.compute_eucledian_gradients = theano.function(
            [gdx],
            [],
            updates=updates,
            givens=dict(grad_inps),
            name='compute_eucledian_gradients',
            mode=gpu_mode,
            profile=options['profile'])

        # Step 2. Compile function for Computing Riemannian gradients
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
                    nw_cost, nw_preactiv_out = safe_clone([model.train_cost,
                                                           model.preactiv_out],
                                                          replace)
                    nw_gvs = TT.Lop(nw_preactiv_out, model.params,
                                  TT.Rop(TT.grad(nw_cost, nw_preactiv_out),
                                         model.params, cgv))

                    Gvs = [ogv + ngv
                           for (ogv, ngv) in zip(gv_args[1:], nw_gvs)]
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



        rvals, updates = krylov_subspace(
            compute_Gv,
            self.gs,
            old_deltas,
            n_dimensions,
            model.params_shape,
            profile=options['profile'],
            device=options['device'])

        gdx = TT.iscalar('gdx')
        grad_inps = zip(loc_inputs,
                        [x[gdx*options['mbs']:(gdx+1)*options['mbs']] for x
                         in shared_data])
        updates.update(dict(zip(cfn_subspaces, rvals)))
        self.update_krylov_subspace = theano.function(
            [gdx],
            [],
            updates=updates,
            givens=dict(grad_inps),
            profile=options['profile'],
            on_unused_input='warn',
            name='update_krylov_subspace',
            mode=mode)

        alphas = tensor.vector('alphas')
        deltas = []
        nw_params = []
        if options['device'] == 'gpu':
            params = model.params
        else:
            params = model.cpu_params

        for param, subspace in zip(params, cfn_subspaces):
            alpha_reshuffle = [0] + ['x'] * param.ndim
            delta = (alphas.dimshuffle(*alpha_reshuffle) * \
                        subspace).sum(axis=0)
            nw_param = param + delta
            nw_params.append(nw_param)
            deltas.append(delta)

        print 'constructing evaluation function'
        ebdx = TT.iscalar('ebdx')

        updates_dict = dict(zip(model.params + old_deltas,
                                nw_params + deltas))
        if options['device'] != 'gpu':
            updates_dict.update(dict(zip(model.cpu_params, nw_params)))

        self.update_params = theano.function([alphas],
                                             updates = updates_dict,
                                             name='update_params',
                                             allow_input_downcast=True,
                                             mode=mode,
                                             profile=options['profile'])

        n_steps = options['ebs'] // options['cbs']
        def ls_cost_step(_idx, acc):
            idx = TT.cast(_idx, 'int32')
            nw_inps = [x[idx * options['cbs']: \
                         (idx + 1) * options['cbs']] for x in loc_inputs]
            replace = dict(zip(model.inputs + model.params, nw_inps +
                               nw_params))
            nw_cost = safe_clone(model.train_cost, replace=replace)
            return [_idx + const(1),
                    acc + nw_cost]

        states = [TT.constant(numpy.float32([0])),
                  TT.constant(numpy.float32([0]))]
        rvals, _ = scan(ls_cost_step,
                        states = states,
                        n_steps = n_steps,
                        name='ls_cost_step',
                        mode=gpu_mode,
                        profile = options['profile'])
        fcost = rvals[1][0] / const(n_steps)

        def ls_grad_step(_idx, gws):
            idx = TT.cast(_idx, 'int32')
            nw_inps = [x[idx * options['cbs']: (idx + 1) * options['cbs']]
                       for x in loc_inputs]
            replace = dict(zip(model.inputs + model.params, nw_inps +
                               nw_params))
            nw_cost = safe_clone(model.train_cost, replace=replace)
            nw_gs = TT.grad(nw_cost, alphas)
            return _idx + numpy.float32(1), gws + nw_gs

        states = [TT.constant(numpy.float32([0])),
                  TT.constant(numpy.zeros((1, n_dimensions),dtype='float32'))]
        rvals, _ = scan(ls_grad_step,
                        states = states,
                        n_steps = n_steps,
                        name = 'ls_grad_step',
                        mode = gpu_mode,
                        profile=options['profile'])

        fgrad = rvals[1][0] / const(n_steps)

        grad_inps = zip(loc_inputs,
                        [x[ebdx*options['ebs']:(ebdx+1)*options['ebs']] for x
                         in shared_data])
        self.lbfgs_fn = theano.function([alphas, ebdx],
                                   #theano.printing.Print('fcost')(fcost),
                                    fcost,
                                   givens=grad_inps,
                                   allow_input_downcast=True,
                                   on_unused_input='warn',
                                   name='lbfgs_fn',
                                   profile=options['profile'],
                                   mode=gpu_mode)
        self.lbfgs_grad = theano.function([alphas, ebdx],
                                     fgrad,
                                     givens=grad_inps,
                                     on_unused_input='warn',
                                     allow_input_downcast=True,
                                     name='lbfgs_grad',
                                     profile=options['profile'],
                                     mode=gpu_mode)

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
        self.compute_error = theano.function([],
                           ferr,
                           givens=dict(zip(loc_inputs, shared_data)),
                           name='compute_err',
                           mode=gpu_mode,
                           on_unused_input='warn',
                           profile=options['profile'])

    def __call__(self):
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
        #self.xdata.set_value(self.data['train_x'][self.permg[self.posg]*gbs:
        #                                          (self.permg[self.posg]+1)*gbs])
        #self.ydata.set_value(self.data['train_y'][self.permg[self.posg]*gbs:
        #                                          (self.permg[self.posg]+1)*gbs])
        print 'Computing eucledian'
        self.compute_eucledian_gradients(self.permg[self.posg])
        g_ed = time.time()
        r_st = time.time()
        #self.xdata.set_value(self.data['train_x'][self.permr[self.posr]*mbs:
        #                                          (self.permr[self.posr]+1)*mbs])
        #self.ydata.set_value(self.data['train_y'][self.permr[self.posr]*mbs:
        #                                          (self.permr[self.posr]+1)*mbs])
        print 'Krylov subspace'
        self.update_krylov_subspace(self.permr[self.posr])
        r_ed = time.time()
        e_st = time.time()
        #self.xdata.set_value(self.data['train_x'][self.perme[self.pose]*ebs:
        #                                          (self.perme[self.pose]+1)*ebs])
        #self.ydata.set_value(self.data['train_y'][self.perme[self.pose]*ebs:
        #                                          (self.perme[self.pose]+1)*ebs])
        a0 = numpy.zeros((self.n_dimensions,),
                         dtype=theano.config.floatX)
        print 'Getting into scipy'
        rvals = scipy.optimize.fmin_bfgs(
            self.lbfgs_fn,
            a0,
            self.lbfgs_grad,
            args=(self.perme[self.pose],),
            maxiter=self.options['lbfgsIters'],
            full_output=True,
            disp=False)
        self.update_params(rvals[0])
        e_ed = time.time()

        st = time.time()
        error = self.compute_error()
        comp_error_time = time.time() - st

        if self.verbose > 1:
            if rvals[-1] == 0 :
                msg = (' .. BFGS finished successfully with solutions %8.5f '
                       'using %03d func calls and %03d grad calls in '
                       '%5.2f sec')
            elif rvals[-1] == 1:
                msg = (' .. BFGS exceeded max number of iterations. Found '
                       'solution is %8.5f using %03d func calls and %03d '
                       'grad calls in %5.2f sec')
            else:
                msg = (' .. BFGS finished due to loss of precision. Found '
                       'solution is %8.5f using %03d func calls and %03d '
                       'grad calls in %5.2f sec')

            print msg % (
                rvals[1], rvals[4], rvals[5], time.time() - e_st)
        if self.verbose > 0:
            msg = ('.. iter %4d score %8.5f, step_size %12.9f '
                   'error %12.9f time [grad] %s,'
                   '[krylov_subspace] %s,'
                   '[updates param] %s,'
                   '%2d(%2d/%2d) [bsg] %2d(%2d/%2d) [bsr] %2d(%2d/%2d)[bse]')
            print msg % (
                self.k,
                rvals[1],
                numpy.sqrt(numpy.sum(rvals[0]**2)),
                error,
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
            'score': rvals[1],
            'error': error,
            'time_err' : comp_error_time,
            'time_grads': g_ed - g_st,
            'time_metric': r_ed - r_st,
            'time_eval': e_ed - e_st,
            'minres_flag': numpy.nan,
            'minres_iters': numpy.nan,
            'minres_relres': numpy.nan,
            'minres_Anorm': numpy.nan,
            'minres_Acond': numpy.nan,
            'grad_norm': numpy.nan,
            'beta': 0,
            'damping': 0,
            'rho' : 0,
            'lambda': numpy.float32(0)}
        return ret

def print_time(secs):
    if secs < 120.:
        return '%6.3f sec' % secs
    elif secs <= 60*60:
        return '%6.3f min' % (secs / 60.)
    else:
        return '%6.3f h  ' % (secs / 3600.)

