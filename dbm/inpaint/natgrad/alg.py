from __future__ import division
"""
Hack to train inpainting DBMs with Razvan's natural gradient optimizer
Everything here is thoroughly hacky and specialized to the inpainting DBMs
Also specialized for binary variables
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow, David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow, David Warde-Farley"
__email__ = "goodfeli@iro"
import warnings
from theano import function
from theano import config
import numpy as np
from theano import tensor as T
from pylearn2.monitor import Monitor
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.utils import sharedX
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils.iteration import is_stochastic
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils.timing import log_timing
from theano.gof.op import get_debug_values
from for_Ian import ThingForIan
import logging
from collections import OrderedDict


log = logging.getLogger(__name__)

class Alg(TrainingAlgorithm):
    """
    Hacky algorithm for training inpainting DBMs with binary inputs and
    softmax outputs using Razvan's natural gradient code
    """
    def __init__(self, learning_rate = .1, cost=None, batch_size=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 termination_criterion=None, update_callbacks=None,
                 init_momentum = None, set_batch_size = False,
                 train_iteration_mode = None, batches_per_iter=None,
                 theano_function_mode = None, monitoring_costs=None,
                 seed = None):
        """
            WRITEME

            learning_rate: The learning rate to use.
                            Train object callbacks can change the learning
                            rate after each epoch. SGD update_callbacks
                            can change it after each minibatch.
            cost: a pylearn2.costs.cost.Cost object specifying the objective
                  function to be minimized.
                  Optionally, may be None. In this case, SGD will call the model's
                  get_default_cost method to obtain the objective function.
            init_momentum: if None, does not use momentum
                            otherwise, use momentum and initialize the
                            momentum coefficient to init_momentum.
                            Callbacks can change this over time just like
                            the learning rate.

                            If the gradient is the same on every step, then
                            the update taken by the SGD algorithm is scaled
                            by a factor of 1/(1-momentum).

                            See section 9 of Geoffrey Hinton's "A Practical
                            Guide to Training Restricted Boltzmann Machines"
                            for details.
            set_batch_size: if True, and batch_size conflicts with
                            model.force_batch_size, will call
                            model.set_batch_size(batch_size) in an attempt
                            to change model.force_batch_size
            theano_function_mode: The theano mode to compile the updates function with.
                            Note that pylearn2 includes some wraplinker modes that are
                            not bundled with theano. See pylearn2.devtools. These
                            extra modes let you do things like check for NaNs at every
                            step, or record md5 digests of all computations performed
                            by the update function to help isolate problems with nondeterminism.

            Parameters are updated by the formula:

            inc := momentum * inc - learning_rate * d cost / d param
            param := param + inc
        """

        if seed is None:
            seed = [2013, 4, 11]

        if isinstance(cost, (list, tuple, set)):
            raise TypeError("SGD no longer supports using collections of Costs to represent "
                    " a sum of Costs. Use pylearn2.costs.cost.SumOfCosts instead.")

        self.learning_rate = learning_rate
        self.cost = cost
        self.batch_size = batch_size
        self.set_batch_size = set_batch_size
        self.batches_per_iter = batches_per_iter
        self._set_monitoring_dataset(monitoring_dataset)
        self.monitoring_batches = monitoring_batches
        if monitoring_dataset is None:
            if monitoring_batches is not None:
                raise ValueError("Specified an amount of monitoring batches but not a monitoring dataset.")
        self.termination_criterion = termination_criterion
        self.init_momenutm = init_momentum
        if init_momentum is None:
            self.momentum = None
        else:
            assert init_momentum >= 0.
            assert init_momentum < 1.
            self.momentum = sharedX(init_momentum, 'momentum')
        self._register_update_callbacks(update_callbacks)
        if train_iteration_mode is None:
            train_iteration_mode = 'shuffled_sequential'
        self.train_iteration_mode = train_iteration_mode
        self.first = True
        self.rng = np.random.RandomState(seed)
        self.theano_function_mode = theano_function_mode
        self.monitoring_costs = monitoring_costs

    def setup(self, model, dataset):

        if self.cost is None:
            self.cost = model.get_default_cost()

        inf_params = [ param for param in model.get_params() if np.any(np.isinf(param.get_value())) ]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: "+str(inf_params))
        if any([np.any(np.isnan(param.get_value())) for param in model.get_params()]):
            nan_params = [ param for param in model.get_params() if np.any(np.isnan(param.get_value())) ]
            raise ValueError("These params are NaN: "+str(nan_params))
        self.model = model

        batch_size = self.batch_size
        if hasattr(model, "force_batch_size"):
            if model.force_batch_size > 0:
                if batch_size is not None:
                    if batch_size != model.force_batch_size:
                        if self.set_batch_size:
                            model.set_batch_size(batch_size)
                        else:
                            raise ValueError("batch_size argument to SGD conflicts with model's force_batch_size attribute")
                else:
                    self.batch_size = model.force_batch_size
        model._test_batch_size = self.batch_size
        self.monitor = Monitor.get_monitor(model)
        # TODO: come up with some standard scheme for associating training runs
        # with monitors / pushing the monitor automatically, instead of just
        # enforcing that people have called push_monitor
        assert self.monitor.get_examples_seen() == 0
        self.monitor._sanity_check()




        X = model.get_input_space().make_theano_batch(name="%s[X]" % self.__class__.__name__)
        self.topo = not X.ndim == 2

        if config.compute_test_value == 'raise':
            if self.topo:
                X.tag.test_value = dataset.get_batch_topo(self.batch_size)
            else:
                X.tag.test_value = dataset.get_batch_design(self.batch_size)

        Y = T.matrix(name="%s[Y]" % self.__class__.__name__)

        supervised = self.cost.supervised

        warnings.warn("Add interface for ThingForIan's hyperparams to Alg")

        self.thing = ThingForIan(
                 X,
                 Y,
                 lr = self.learning_rate,
                 dbm = model,
                 cost = self.cost,
                 batchsize=self.batch_size,
                 init_damp = 5.,
                 min_damp = .001,
                 damp_ratio = 5./4.,
                 mrtol = 1e-4,
                 miters = 100,
                 trancond = 1e-4,
                 adapt_rho = 1)

        if supervised:
            if config.compute_test_value == 'raise':
                _, Y.tag.test_value = dataset.get_batch_design(self.batch_size, True)

            self.supervised = True
            cost_value = self.thing.cost(model, X, Y)

        else:
            raise NotImplementedError()
            self.supervised = False
            cost_value = self.cost(model, X)
        if cost_value is not None and cost_value.name is None:
            if self.supervised:
                cost_value.name = 'objective(' + X.name + ', ' + Y.name + ')'
            else:
                cost_value.name = 'objective(' + X.name + ')'

        # Set up monitor to model the objective value, learning rate,
        # momentum (if applicable), and extra channels defined by
        # the cost
        learning_rate = self.learning_rate
        if self.monitoring_dataset is not None:
            self.monitor.setup(dataset=self.monitoring_dataset,
                    cost=self.cost, batch_size=self.batch_size, num_batches=self.monitoring_batches,
                    extra_costs=self.monitoring_costs
                    )
            if self.supervised:
                ipt = (X, Y)
            else:
                ipt = X
            dataset_name = self.monitoring_dataset.keys()[0]
            monitoring_dataset = self.monitoring_dataset[dataset_name]
            #TODO: have Monitor support non-data-dependent channels
            self.monitor.add_channel(name='learning_rate', ipt=ipt,
                    val=learning_rate, dataset=monitoring_dataset)
            if self.momentum:
                self.monitor.add_channel(name='momentum', ipt=ipt,
                        val=self.momentum, dataset=monitoring_dataset)

        params = list(model.get_params())
        assert len(params) > 0
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i

        if self.cost.supervised:
            grads, updates = self.cost.get_gradients(model, X, Y)
        else:
            grads, updates = self.cost.get_gradients(model, X)

        for param in grads:
            assert param in params
        for param in params:
            assert param in grads

        for param in grads:
            if grads[param].name is None and cost_value is not None:
                grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                     {'costname': cost_value.name,
                                      'paramname': param.name})

        lr_scalers = model.get_lr_scalers()

        for key in lr_scalers:
            if key not in params:
                raise ValueError("Tried to scale the learning rate on " +\
                        str(key)+" which is not an optimization parameter.")

        log.info('Parameter and initial learning rate summary:')
        for param in params:
            param_name = param.name
            if param_name is None:
                param_name = 'anon_param'
            lr = learning_rate * lr_scalers.get(param,1.)
            log.info('\t' + param_name + ': ' + str(lr))

        if self.momentum is None:
            updates.update( dict(safe_zip(params, [param - learning_rate * \
                lr_scalers.get(param, 1.) * grads[param]
                                    for param in params])))
        else:
            for param in params:
                inc = sharedX(param.get_value() * 0.)
                if param.name is not None:
                    inc.name = 'inc_'+param.name
                updated_inc = self.momentum * inc - learning_rate * lr_scalers.get(param, 1.) * grads[param]
                updates[inc] = updated_inc
                updates[param] = param + updated_inc


        for param in params:
            if updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'
        model.censor_updates(updates)
        for param in params:
            update = updates[param]
            if update.name is None:
                update.name = 'censor(sgd_update(' + param.name + '))'
            for update_val in get_debug_values(update):
                if np.any(np.isinf(update_val)):
                    raise ValueError("debug value of %s contains infs" % update.name)
                if np.any(np.isnan(update_val)):
                    raise ValueError("debug value of %s contains nans" % update.name)


        with log_timing(log, 'Compiling sgd_update'):
            if self.supervised:
                fn_inputs = [X, Y]
            else:
                fn_inputs = [X]
            self.sgd_update = function(fn_inputs, updates=updates,
                                       name='sgd_update',
                                       on_unused_input='ignore',
                                       mode=self.theano_function_mode)
        self.params = params

    def train(self, dataset):
        if not hasattr(self, 'sgd_update'):
            raise Exception("train called without first calling setup")
        model = self.model
        batch_size = self.batch_size

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        self.first = False
        rng = self.rng
        if not is_stochastic(self.train_iteration_mode):
            rng = None
        iterator = dataset.iterator(mode=self.train_iteration_mode,
                batch_size=self.batch_size, targets=self.supervised,
                topo=self.topo, rng = rng, num_batches = self.batches_per_iter)
        if self.topo:
            batch_idx = dataset.get_topo_batch_axis()
        else:
            batch_idx = 0
        if self.supervised:
            for (batch_in, batch_target) in iterator:
                self.thing.on_load_batch(batch_in, batch_target)
                self.thing.step(batch_in, batch_target)

                actual_batch_size = batch_in.shape[batch_idx]
                self.monitor.report_batch(actual_batch_size)
                for callback in self.update_callbacks:
                    callback(self)
        else:
            raise NotImplementedError()
            for batch in iterator:
                self.sgd_update(batch)
                actual_batch_size = batch.shape[0] # iterator might return a smaller batch if dataset size
                                                   # isn't divisible by batch_size
                self.monitor.report_batch(actual_batch_size)
                for callback in self.update_callbacks:
                    callback(self)

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

    def continue_learning(self, model):
        if self.termination_criterion is None:
            return True
        else:
            return self.termination_criterion.continue_learning(self.model)


