import numpy
import time

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan

from minres import minres, minres_messages
from minres import minresQLP, minresQLP_messages
from utils import forloop, safe_clone, print_time, print_mem, const


class natSGD(object):
    def __init__(self,
                 model,
                 state,
                 data):
        """
        Parameters:
            :param model:
                Class describing the model used.  It should provide the
                 computational graph to evaluate the model
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
        """

        #####################################
        # Step 0. Constructs shared variables
        #####################################
        n_params = len(model.params)
        cbs = state['cbs']
        bs = state['bs']
        ebs = state['ebs']
        mbs = state['mbs']
        profile = state['profile']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        self.damping = theano.shared(numpy.float32(state['damp']))

        self.gs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]
        self.rs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]

        self.loop_inps = [theano.shared(
            numpy.zeros(shp, dtype=theano.config.floatX))
            for shp in model.params_shape]
        self.loop_outs = [theano.shared(
            numpy.zeros(shp, dtype=theano.config.floatX))
            for shp in model.params_shape]
        self.step = 0
        self.cbs = cbs
        self.bs = bs
        self.ebs = ebs
        self.mbs = mbs
        self.state = state
        self.profile = profile
        self.data = data
        self.step_timer = time.time()

        ############################################################
        # Step 1. Compile function for computing eucledian gradients
        ############################################################
        print 'Constructing grad function'
        bdx = TT.iscalar('batch_idx')
        loc_data = [x[bdx * cbs: (bdx + 1) * cbs] for x in
                    self.data.variables]
        cost = safe_clone(model.train_cost, model.inputs, loc_data)
        gs = TT.grad(cost, model.params)
        ratio = numpy.float32(float(bs) / cbs)
        update = [(g, g + lg / ratio) for g, lg in zip(self.gs, gs)]

        print 'Compiling grad function'
        st = time.time()
        self.loc_grad_fn = theano.function(
            [bdx ],
            [],
            updates=update, name='loc_fn_grad',
            profile=profile)
        print 'took', time.time() - st

        #############################################################
        # Step 2. Compile function for Computing Riemannian gradients
        #############################################################
        loc_x = self.data._natgrad[bdx*cbs: (bdx+1)*cbs]
        loc_y = self.data._natgrady[bdx*cbs:(bdx+1)*cbs]
        loc_Gvs = safe_clone(model.Gvs(*self.loop_inps), [model.X, model.Y],
                             [loc_x, loc_y])
        updates = [(l, l + lg) for l, lg in zip(self.loop_outs, loc_Gvs)]
        st = time.time()
        loc_Gv_fn = theano.function(
            [bdx], [], updates=updates, name='loc_fn_rop', profile=profile)
        print 'took', time.time() - st

        def compute_Gv(*args):
            rval = forloop(loc_Gv_fn,
                           mbs // cbs,
                           self.loop_inps,
                           self.loop_outs)(*args)
            return rval, {}

        print 'Constructing riemannian gradient function'
        st = time.time()
        norm_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in self.gs))
        if not state['minresQLP']:
            self.msgs = minres_messages
            rvals = minres(compute_Gv,
                           [x / norm_grads for x in self.gs],
                           rtol=state['mrtol'],
                           damp=self.damping,
                           maxit=state['miters'],
                           profile=state['profile'])
        else:
            self.msgs = minresQLP_messages[1:]
            rvals = minresQLP(compute_Gv,
                              [x / norm_grads for x in self.gs],
                              model.params_shape,
                              rtol=state['mrtol'],
                              damp=self.damping,
                              maxit=state['miters'],
                              TranCond=state['trancond'],
                              profile=state['profile'])

        nw_rs = [x * norm_grads for x in rvals[0]]
        flag = TT.cast(rvals[1], 'int32')
        niters = rvals[2]
        rel_residual = rvals[3]
        Anorm = rvals[4]
        Acond = rvals[5]

        norm_rs_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in nw_rs))
        norm_ord0 = TT.max(abs(nw_rs[0]))
        for r in nw_rs[1:]:
            norm_ord0 = TT.maximum(norm_ord0,
                                   TT.max(abs(r)))
        updates = zip(self.rs, nw_rs)
        print 'took', time.time() - st
        print 'Compiling riemannian gradient function'
        st = time.time()
        self.compute_natural_gradients = theano.function(
            [],
            [flag, niters, rel_residual, Anorm, Acond,
             norm_grads, norm_rs_grads, norm_ord0],
            updates=updates,
            allow_input_downcast = True,
            name='compute_riemannian_gradients',
            on_unused_input='warn',
            profile=profile)
        print 'took', time.time() - st
        ###########################################################
        # Step 3. Compile function for evaluating cost and updating
        # parameters
        ###########################################################
        print 'constructing evaluation function'
        lr = TT.scalar('lr')
        self.lr = numpy.float32(state['lr'])
        loc_data = [x[bdx * cbs: (bdx + 1) * cbs] for x in
                    self.data.variables]
        old_cost = safe_clone(model.train_cost, model.inputs, loc_data)
        self.loc_old_cost = theano.function(
            [bdx], old_cost, name='loc_old_cost', profile=profile)
        new_params = [p - lr * r for p, r in zip(model.params, self.rs)]
        new_cost = safe_clone(model.train_cost,
                              model.inputs + model.params,
                              loc_data + new_params)
        new_err = safe_clone(model.error,
                             model.inputs + model.params,
                             loc_data + new_params)
        self.loc_new_cost = theano.function(
            [bdx, lr], [new_cost, new_err], name='loc_new_cost',
            profile=profile)

        self.lr = numpy.float32(state['lr'])
        updates = dict(zip(model.params, new_params))
        model.dbm_class.censor_updates(updates)
        self.update_params = theano.function(
            [lr], [], updates=updates,
            name='update_params')
        old_cost = TT.scalar('old_cost')
        new_cost = TT.scalar('new_cost')
        p_norm = TT.scalar('p_norm')
        prod = sum([TT.sum(g * r) for g, r in zip(self.gs, self.rs)])
        #pnorm = TT.sqrt(sum(TT.sum(g*g) for g in self.gs)) * \
        #        TT.sqrt(sum(TT.sum(r*r) for r in self.rs))
        dist = -lr * prod
        angle = prod / p_norm
        rho = (new_cost - old_cost) / dist
        self.compute_rho = theano.function(
            [old_cost, new_cost, lr, p_norm], [rho, dist, angle], name='compute_rho', profile=profile)
        self.old_cost = 1e20
        self.__new_cost = 0
        self.__error = 0
        self.return_names = ['cost',
                             'old_cost',
                             'error',
                             'time_grads',
                             'time_metric',
                             'time_eval',
                             'minres_flag',
                             'minres_iters',
                             'minres_relres',
                             'minres_Anorm',
                             'minres_Acond',
                             'norm_ord0',
                             'norm_grad',
                             'norm_nat',
                             'lr',
                             'grad_angle',
                             #'r_g',
                             #'icost',
                             'damping',
                             'rho'
                            ]

    def compute_gradients(self):
        for g in self.gs:
            g.container.storage[0][:] = 0
        for idx in xrange(self.bs // self.cbs):
             self.loc_grad_fn(idx)
        return None

    def compute_old_cost(self):
        costs = [self.loc_old_cost(idx)
                 for idx in xrange(self.bs // self.cbs)]
        return numpy.mean(costs).astype(theano.config.floatX)

    def compute_new_cost(self, lr):
        rvals = [self.loc_new_cost(idx, self.lr)
                 for idx in xrange(self.bs // self.cbs)]
        cost = numpy.mean([x for x, y in
                            rvals]).astype(theano.config.floatX)
        error = numpy.mean([y for x, y in
                             rvals]).astype(theano.config.floatX)
        return cost, error

    def __call__(self):
        self.data.update_before_computing_gradients()
        g_st = time.time()
        new_cost = self.compute_gradients()
        g_ed = time.time()
        self.data.update_before_computing_natural_gradients()
        r_st = time.time()
        rvals = self.compute_natural_gradients()
        r_ed = time.time()
        self.data.update_before_evaluation()
        e_st = time.time()
        old_cost = self.compute_old_cost()
        new_cost, error = self.compute_new_cost(self.lr)
        rho, r_g, angle = self.compute_rho(old_cost, new_cost, self.lr,
                                           rvals[5]*rvals[6])
        if self.step < 1:
            rho = .6

        if self.state['adapt'] == 1:
            if rho < .25:
                self.damping.set_value(numpy.float32(
                   self.damping.get_value() * self.state['damp_ratio']))
            elif rho > .75 and self.damping.get_value() > self.state['mindamp']:
                self.damping.set_value(numpy.float32(
                        self.damping.get_value() / self.state['damp_ratio']))
        e_ed = time.time()

        if new_cost >= old_cost:
                print ('Variance too large on training cost!')
                self.damping.set_value(numpy.float32(
                    self.damping.get_value() + 1.))
        else:
            self.update_params(self.lr)

        print 'Minres: %s' % self.msgs[rvals[0]], \
                        '# iters %04d' % rvals[1], \
                        'relative error residuals %.4g' % rvals[2], \
                        'Anorm', rvals[3], 'Acond', rvals[4]
        msg = ('.. iter %4d '
               'cost %.4e '
               'old_cost %.4e '
               'step_size %.1e '
               ' damping %.1e '
               'ord0_norm %.1e '
               'norm grad %.1e '
               'norm nat grad %.1e '
               'angle %.1e '
               'rho %.1e '
               'time [grad] %s,'
               '[riemann grad] %s,'
               '[updates param] %s,'
               'whole time %s')
        print msg % (
            self.step,
            new_cost,
            old_cost,
            self.lr,
            self.damping.get_value(),
            rvals[7],
            rvals[5],
            rvals[6],
            angle,
            rho,
            print_time(g_ed - g_st),
            print_time(r_ed - r_st),
            print_time(e_ed - e_st),
            print_time(time.time() - self.step_timer))
        self.step_timer = time.time()

        self.old_cost = new_cost

        e_ed = time.time()


        self.step += 1

        ret = {
            'cost': float(new_cost),
            'old_cost':float(old_cost),
            'error': float(error),
            'time_grads': float(g_ed - g_st),
            'time_metric': float(r_ed - r_st),
            'time_eval': float(e_ed - e_st),
            'minres_flag': rvals[0],
            'minres_iters': rvals[1],
            'minres_relres': rvals[2],
            'minres_Anorm': rvals[3],
            'minres_Acond': rvals[4],
            'norm_ord0': rvals[7],
            'norm_grad':rvals[5],
            'norm_nat': rvals[6],
            'grad_angle' : float(angle),
            'rho': float(rvals[-1]),
            'lr': self.lr,
            #'r_g': float(r_g),
            #'icost' : float(tmp_cost),
            'damping': self.damping.get_value(),
            'rho': numpy.float32(rho)
        }
        return ret
