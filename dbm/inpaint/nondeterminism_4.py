import galatea.dbm.inpaint.super_dbm
import galatea.dbm.inpaint.super_inpaint
from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.datasets.dataset import Dataset
from pylearn2.devtools import disturb_mem
import numpy as np
from pylearn2.monitor import Monitor
from pylearn2.utils import sharedX
from pylearn2.utils import safe_izip
from pylearn2.models.dbm import DBM
from pylearn2.models.dbm import WeightDoubling
from pylearn2.models.dbm import BinaryVector
from pylearn2.models.dbm import BinaryVectorMaxPool
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.costs.cost import Cost

class A(Cost):
    def __init__(self,
                    noise = False,
                    both_directions = False,
                    l1_act_coeffs = None,
                    l1_act_targets = None,
                    l1_act_eps = None,
                    range_rewards = None,
                    stdev_rewards = None,
                    robustness = None,
                    supervised = False,
                    niter = None,
                    block_grad = None,
                    vis_presynaptic_cost = None,
                    hid_presynaptic_cost = None,
                    reweighted_act_coeffs = None,
                    reweighted_act_targets = None,
                    toronto_act_targets = None,
                    toronto_act_coeffs = None
                    ):
        self.__dict__.update(locals())
        del self.self


    def get_monitoring_channels(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None):

        rval = OrderedDict()

        if drop_mask is not None and drop_mask.ndim < X.ndim:
            if X.ndim != 4:
                raise NotImplementedError()
            drop_mask = drop_mask.dimshuffle(0,1,2,'x')

        scratch = self(model, X, Y, drop_mask = drop_mask, drop_mask_Y = drop_mask_Y,
                return_locals = True)

        history = scratch['history']
        new_history = scratch['new_history']
        new_drop_mask = scratch['new_drop_mask']
        new_drop_mask_Y = None
        if self.supervised:
            drop_mask_Y = scratch['drop_mask_Y']
            new_drop_mask_Y = scratch['new_drop_mask_Y']

        for ii, packed in enumerate(safe_izip(history, new_history)):
            state, new_state = packed
            rval['inpaint_after_' + str(ii)] = self.cost_from_states(state,
                    new_state,
                    model, X, Y, drop_mask, drop_mask_Y,
                    new_drop_mask, new_drop_mask_Y)

            if ii > 0:
                prev_state = history[ii-1]
                V_hat = state['V_hat']
                prev_V_hat = prev_state['V_hat']
                rval['max_pixel_diff[%d]'%ii] = abs(V_hat-prev_V_hat).max()

        final_state = history[-1]

        layers = [ model.visible_layer ] + model.hidden_layers
        states = [ final_state['V_hat'] ] + final_state['H_hat']

        for layer, state in safe_izip(layers, states):
            d = layer.get_monitoring_channels_from_state(state)
            for key in d:
                mod_key = 'final_inpaint_' + layer.layer_name + '_' + key
                assert mod_key not in rval
                rval[mod_key] = d[key]

        return rval

    def __call__(self, model, X, Y = None, drop_mask = None, drop_mask_Y = None,
            return_locals = False, include_toronto = True):

        if not self.supervised:
            assert drop_mask_Y is None
            Y = None # ignore Y if some other cost is supervised and has made it get passed in

        dbm = model

        if drop_mask.ndim < X.ndim:
            if X.ndim != 4:
                raise NotImplementedError()
            drop_mask = drop_mask.dimshuffle(0,1,2,'x')

        history = dbm.do_inpainting(X, drop_mask = drop_mask)
        final_state = history[-1]

        new_drop_mask = None
        new_drop_mask_Y = None
        new_history = [ None for state in history ]

        new_final_state = new_history[-1]

        total_cost = self.cost_from_states(final_state, new_final_state, dbm, X, Y, drop_mask, drop_mask_Y, new_drop_mask, new_drop_mask_Y)

        if return_locals:
            return locals()

        return total_cost



    def get_gradients(self, model, X, Y = None, **kwargs):
        assert False

        scratch = self(model, X, Y, include_toronto = False, return_locals=True, **kwargs)

        total_cost = scratch['total_cost']

        params = list(model.get_params())
        grads = dict(safe_zip(params, T.grad(total_cost, params, disconnected_inputs='ignore')))

        if self.toronto_act_targets is not None:
            H_hat = scratch['history'][-1]['H_hat']
            for i, packed in enumerate(safe_zip(H_hat, self.toronto_act_coeffs, self.toronto_act_targets)):
                s, c, t = packed
                if c == 0.:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                m_cost = c * T.sqr(m-t).mean()
                real_grads = T.grad(m_cost, s)
                if i == 0:
                    below = X
                else:
                    below = H_hat[i-1][0]
                W, = model.hidden_layers[i].transformer.get_params()
                assert W in grads
                b = model.hidden_layers[i].b

                ancestor = T.scalar()
                hack_W = W + ancestor
                hack_b = b + ancestor

                fake_s = T.dot(below, hack_W) + hack_b
                if fake_s.ndim != real_grads.ndim:
                    print fake_s.ndim
                    print real_grads.ndim
                    assert False
                sources = [ (fake_s, real_grads) ]

                fake_grads = T.grad(cost=None, known_grads=dict(sources), wrt=[below, ancestor, hack_W, hack_b])

                grads[W] = grads[W] + fake_grads[2]
                grads[b] = grads[b] + fake_grads[3]


        return grads, OrderedDict()


    def cost_from_states(self, state, new_state, dbm, X, Y, drop_mask, drop_mask_Y,
            new_drop_mask, new_drop_mask_Y):

        assert drop_mask_Y is None
        assert new_drop_mask_Y is None

        V_hat_unmasked = state['V_hat_unmasked']
        assert V_hat_unmasked.ndim == X.ndim

        inpaint_cost = V_hat_unmasked.sum()
        return inpaint_cost


        if new_state is not None:

            new_V_hat_unmasked = new_state['V_hat_unmasked']

            new_inpaint_cost = dbm.visible_layer.recons_cost(X, new_V_hat_unmasked, new_drop_mask)
            if self.supervised:
                new_Y_hat_unmasked = new_state['Y_hat_unmasked']
                new_inpaint_cost = new_inpaint_cost + \
                        dbm.hidden_layers[-1].recons_cost(Y, new_Y_hat_unmasked, new_drop_mask_Y, scale)
            # end if include_Y
            inpaint_cost = 0.5 * inpaint_cost + 0.5 * new_inpaint_cost
        # end if both directions

        total_cost = inpaint_cost

        if self.range_rewards is not None:
            for layer, mf_state, coeffs in safe_izip(
                    dbm.hidden_layers,
                    state['H_hat'],
                    self.range_rewards):
                try:
                    layer_cost = layer.get_range_rewards(mf_state, coeffs)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    total_cost += layer_cost

        if self.stdev_rewards is not None:
            for layer, mf_state, coeffs in safe_izip(
                    dbm.hidden_layers,
                    state['H_hat'],
                    self.stdev_rewards):
                try:
                    layer_cost = layer.get_stdev_rewards(mf_state, coeffs)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    total_cost += layer_cost

        if self.l1_act_targets is not None:
            if self.l1_act_eps is None:
                self.l1_act_eps = [ None ] * len(self.l1_act_targets)
            for layer, mf_state, targets, coeffs, eps in safe_izip(dbm.hidden_layers, state['H_hat'] , self.l1_act_targets, self.l1_act_coeffs, self.l1_act_eps):
                assert not isinstance(targets, str)

                try:
                    layer_cost = layer.get_l1_act_cost(mf_state, targets, coeffs, eps)
                except NotImplementedError:
                    if coeffs == 0.:
                        layer_cost = 0.
                    else:
                        raise
                if layer_cost != 0.:
                    total_cost += layer_cost
                #for H, t, c in zip(mf_state, targets, coeffs):
                    #if c == 0.:
                    #    continue
                    #axes = (0,2,3) # all but channel axis
                                  # this assumes bc01 format
                    #h = H.mean(axis=axes)
                    #assert h.ndim == 1
                    #total_cost += c * abs(h - t).mean()
                # end for substates
            # end for layers
        # end if act penalty

        if self.hid_presynaptic_cost is not None:
            for c, s, in safe_izip(self.hid_presynaptic_cost, state['H_hat']):
                if c == 0.:
                    continue
                s = s[1]
                assert hasattr(s, 'owner')
                owner = s.owner
                assert owner is not None
                op = owner.op

                if not hasattr(op, 'scalar_op'):
                    raise ValueError("Expected V_hat_unmasked to be generated by an Elemwise op, got "+str(op)+" of type "+str(type(op)))
                assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
                z ,= owner.inputs

                total_cost += c * T.sqr(z).mean()

        if self.reweighted_act_targets is not None:
            # hardcoded for sigmoid layers
            for c, t, s in safe_izip(self.reweighted_act_coeffs, self.reweighted_act_targets, state['H_hat']):
                if c == 0:
                    continue
                s, _ = s
                m = s.mean(axis=0)
                d = T.sqr(m-t)
                weight = 1./(1e-7+s*(1-s))
                total_cost += c * (weight * d).mean()


        total_cost.name = 'total_cost(V_hat_unmasked = %s)' % V_hat_unmasked.name

        return total_cost

class BinaryVisLayer(BinaryVector):
    def recons_cost(self, V, V_hat_unmasked, drop_mask = None):
        return V_hat_unmasked.sum()

class SuperWeightDoubling(WeightDoubling):
    def do_inpainting(self, V, drop_mask = None):
        dbm = self.dbm

        history = []

        V_hat = V
        V_hat_unmasked = V

        H_hat = []
        H_hat.append(dbm.hidden_layers[0].mf_update(
            state_above = None,
            state_below = V_hat,
            iter_name = '0'))

        def update_history():
            assert V_hat_unmasked.ndim > 1
            d =  { 'V_hat' :  V_hat, 'H_hat' : H_hat, 'V_hat_unmasked' : V_hat_unmasked }
            history.append( d )

        update_history()

        V_hat_unmasked = dbm.hidden_layers[0].downward_message(H_hat[0][0])
        V_hat = V_hat_unmasked
        V_hat.name = 'V_hat[%d](V_hat = %s)' % (1, V_hat.name)

        update_history()

        return history

class ADBM(DBM):
    def setup_inference_procedure(self):
        if not hasattr(self, 'inference_procedure') or \
                self.inference_procedure is None:
            self.inference_procedure = SuperWeightDoubling()
            self.inference_procedure.set_dbm(self)

    def do_inpainting(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.do_inpainting(*args, **kwargs)

    def mf(self, *args, **kwargs):
        self.setup_inference_procedure()
        return self.inference_procedure.mf(*args, **kwargs)

def prereq(*args):
    disturb_mem.disturb_mem()

class InpaintAlgorithm(object):
    def __init__(self, cost, batch_size=None, batches_per_iter=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 max_iter = 5, suicide = False, init_alpha = None,
                 reset_alpha = True, conjugate = False, reset_conjugate = True,
                 termination_criterion = None, set_batch_size = False,
                 line_search_mode = None, min_init_alpha = 1e-3,
                 duplicate = 1, combine_batches = 1, scale_step = 1.,
                 theano_function_mode=None):

        self.__dict__.update(locals())
        if isinstance(monitoring_dataset, Dataset):
            self.monitoring_dataset = { '': monitoring_dataset }

    def setup(self, model, dataset):
        if self.set_batch_size:
            model.set_batch_size(self.batch_size)

        if self.batch_size is None:
            self.batch_size = model.force_batch_size

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_theano_function_mode(self.theano_function_mode)

        space = model.get_input_space()
        X = sharedX( space.get_origin_batch(model.batch_size) , 'BGD_X')
        self.space = space
        rng = np.random.RandomState([2012,7,20])
        test_mask = space.get_origin_batch(model.batch_size)
        test_mask = rng.randint(0,2,test_mask.shape)
        drop_mask = sharedX( np.cast[X.dtype] ( test_mask), name = 'drop_mask')
        self.drop_mask = drop_mask
        assert drop_mask.ndim == test_mask.ndim

        Y = None
        drop_mask_Y = None

        obj = self.cost(model,X, Y, drop_mask = drop_mask, drop_mask_Y = drop_mask_Y)


        if self.monitoring_dataset is not None:
            if not any([dataset.has_targets() for dataset in self.monitoring_dataset.values()]):
                Y = None
            assert X.name is not None
            channels = model.get_monitoring_channels(X,Y)
            assert X.name is not None
            cost_channels = self.cost.get_monitoring_channels(model, X = X, Y = Y, drop_mask = drop_mask,
                    drop_mask_Y = drop_mask_Y)
            for key in cost_channels:
                channels[key] = cost_channels[key]

            for dataset_name in self.monitoring_dataset:

                monitoring_dataset = self.monitoring_dataset[dataset_name]
                self.monitor.add_dataset(dataset=monitoring_dataset,
                                    mode="sequential",
                                    batch_size=self.batch_size,
                                    num_batches=self.monitoring_batches)
                ipt = X
                if Y is not None:
                    ipt = [X,Y]
                self.monitor.add_channel('objective',ipt=ipt,val=obj, dataset=monitoring_dataset, prereqs =  [ prereq ])

                for name in channels:
                    J = channels[name]
                    if isinstance(J, tuple):
                        assert len(J) == 2
                        J, prereqs = J
                    else:
                        prereqs = []

                    prereqs = list(prereqs)
                    prereqs.append(prereq)

                    if Y is not None:
                        ipt = (X,Y)
                    else:
                        ipt = X

                    self.monitor.add_channel(name=name,
                                             ipt=ipt,
                                             val=J, dataset=monitoring_dataset,
                                             prereqs=prereqs)


        self.inputs = None

        self.X = X

def run(replay):
    X = np.zeros((2,2))
    X[0,0] = 1.
    raw_train = DenseDesignMatrix(X=X)

    train = raw_train

    model = ADBM(
            batch_size = 2,
            niter= 2,
            visible_layer= BinaryVector(
                nvis= 2,
                bias_from_marginals = raw_train,
            ),
            hidden_layers= [
                # removing this removes the bug. not sure if I just need to disturb mem though
                BinaryVectorMaxPool(
                    detector_layer_dim= 2,
                            pool_size= 1,
                            sparse_init= 1,
                            layer_name= 'h0',
                            init_bias= 0.
                   )
                  ]
        )
    disturb_mem.disturb_mem()

    algorithm = InpaintAlgorithm(
        theano_function_mode = RecordMode(
                            file_path= "nondeterminism_4.txt",
                            replay=replay
                   ),
                   monitoring_dataset = OrderedDict([
                            ('train', train)
                            ]
                   ),
                   cost= A(),
            )

    algorithm.setup(model=model, dataset=train)
    model.monitor()

    algorithm.theano_function_mode.record.f.flush()
    algorithm.theano_function_mode.record.f.close()

run(0)
run(1)
