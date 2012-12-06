from pylearn2.monitor import Monitor
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
import theano.tensor as T
from theano import function
from pylearn2.utils import sharedX
import numpy as np
import warnings
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.iteration import is_stochastic
from theano import config
from pylearn2.train_extensions import TrainExtension
from pylearn2.termination_criteria import TerminationCriterion
from pylearn2.utils import safe_zip
from pylearn2.models.dbm import flatten
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import OrderedDict
import theano

class SetupBatch:
    def __init__(self,alg):
        self.alg = alg

    def __call__(self, * args):
        if len(args) > 1:
            X, Y = args
            self.alg.setup_batch(X, Y)
        else:
            X = args
            self.alg.setup_batch(X)

    def __getstate__(self):
        return {}

class InpaintAlgorithm(object):
    def __init__(self, mask_gen, cost, batch_size=None, batches_per_iter=None,
                 monitoring_batches=None, monitoring_dataset=None,
                 max_iter = 5, suicide = False, init_alpha = None,
                 reset_alpha = True, conjugate = False, reset_conjugate = True,
                 termination_criterion = None, set_batch_size = False,
                 line_search_mode = None, min_init_alpha = 1e-3,
                 duplicate = 1, combine_batches = 1, scale_step = 1.,
                 theano_function_mode=None):
        assert False # deprecated
        """
        if batch_size is None, reverts to the force_batch_size field of the
        model
        """

        if line_search_mode is None and init_alpha is None:
            init_alpha = ( .001, .005, .01, .05, .1 )

        self.__dict__.update(locals())
        del self.self
        if monitoring_dataset is None:
            assert monitoring_batches == None
        if isinstance(monitoring_dataset, Dataset):
            self.monitoring_dataset = { '': monitoring_dataset }
        self.bSetup = False
        self.rng = np.random.RandomState([2012,10,17])

    def setup_batch(self, X, Y = None):
        assert not isinstance(X,tuple)
        self.X.set_value(X)
        assert self.cost.supervised == (Y is not None)
        if Y is not None:
            assert Y.ndim == 2
            assert self.Y.ndim == 2
            self.Y.set_value(Y)
        self.update_mask()

    def get_setup_batch_object(self):
        return SetupBatch(self)

    def setup(self, model, dataset):
        """
        Allows the training algorithm to do some preliminary configuration
        *before* we actually start training the model. The dataset is provided
        in case other derived training algorithms need to modify model based on
        the dataset.

        Parameters
        ----------
        model: a Python object representing the model to train loosely
        implementing the interface of models.model.Model.

        dataset: a pylearn2.datasets.dataset.Dataset object used to draw
        training data
        """
        self.model = model

        if self.set_batch_size:
            model.set_batch_size(self.batch_size)

        if self.batch_size is None:
            self.batch_size = model.force_batch_size

        model.cost = self.cost
        model.mask_gen = self.mask_gen

        self.monitor = Monitor.get_monitor(model)
        self.monitor.set_theano_function_mode(self.theano_function_mode)
        prereq = self.get_setup_batch_object()
        #We want to use big batches. We need to make several theano calls on each
        #batch. To avoid paying the GPU latency every time, we use a shared variable
        #but the shared variable needs to stay allocated during the time that the
        #monitor is working, and we don't want the monitor to increase the memory
        #overhead. So we make the monitor work off of the same shared variable
        space = model.get_input_space()
        X = sharedX( space.get_origin_batch(model.batch_size) , 'BGD_X')
        self.space = space
        rng = np.random.RandomState([2012,7,20])
        test_mask = space.get_origin_batch(model.batch_size)
        test_mask = rng.randint(0,2,test_mask.shape)
        if hasattr(self.mask_gen,'sync_channels') and self.mask_gen.sync_channels:
            if test_mask.ndim != 4:
                raise NotImplementedError()
            test_mask = test_mask[:,:,:,0]
            assert test_mask.ndim == 3
        drop_mask = sharedX( np.cast[X.dtype] ( test_mask), name = 'drop_mask')
        self.drop_mask = drop_mask
        assert drop_mask.ndim == test_mask.ndim

        Y = None
        drop_mask_Y = None
        if self.cost.supervised:
            Y = sharedX(model.get_output_space().get_origin_batch(model.batch_size), 'BGD_Y')
            self.Y = Y
            test_mask_Y = rng.randint(0,2,(model.batch_size,))
            drop_mask_Y = sharedX( np.cast[Y.dtype](test_mask_Y), name = 'drop_mask_Y')
            self.drop_mask_Y = drop_mask_Y
            dmx, dmy = self.mask_gen(X, Y)
            updates = OrderedDict([ (drop_mask, dmx),\
                    (drop_mask_Y, dmy)] )
        else:
            updates = OrderedDict([( drop_mask, self.mask_gen(X) )])


        obj = self.cost(model,X, Y, drop_mask = drop_mask, drop_mask_Y = drop_mask_Y)
        gradients, gradient_updates = self.cost.get_gradients(model, X, Y, drop_mask = drop_mask,
                drop_mask_Y = drop_mask_Y)

        if hasattr(model.inference_procedure, 'V_dropout'):
            include_prob = model.inference_procedure.include_prob
            theano_rng = MRG_RandomStreams(2012+11+20)
            for elem in flatten([model.inference_procedure.V_dropout, model.inference_procedure.H_dropout]):
                updates[elem] =  theano_rng.binomial(p=include_prob, size=elem.shape, dtype=elem.dtype, n=1) / include_prob
        self.update_mask = function([], updates = updates)


        if self.monitoring_dataset is not None:
            if not any([dataset.has_targets() for dataset in self.monitoring_dataset.values()]):
                Y = None
            assert X.name is not None
            channels = model.get_monitoring_channels(X,Y)
            if not isinstance(channels, dict):
                raise TypeError("model.get_monitoring_channels must return a "
                                "dictionary, but it returned " + str(channels))
            assert X.name is not None
            wtf = self.cost.get_monitoring_channels(model, X = X, Y = Y, drop_mask = drop_mask,
                    drop_mask_Y = drop_mask_Y)
            for key in wtf:
                channels[key] = wtf[key]

            for dataset_name in self.monitoring_dataset:

                if dataset_name == '':
                    prefix = ''
                else:
                    prefix = dataset_name + '_'

                monitoring_dataset = self.monitoring_dataset[dataset_name]
                self.monitor.add_dataset(dataset=monitoring_dataset,
                                    mode="sequential",
                                    batch_size=self.batch_size,
                                    num_batches=self.monitoring_batches)
                #we only need to put the prereq in once to make sure it gets run
                #adding it more times shouldn't hurt, but be careful
                #each time you say "self.setup_batch" you get a new object with a
                #different id, and if you install n of those the prereq will run n
                #times. It won't cause any wrong results, just a big slowdown
                warnings.warn("This is weird-- ipt=(X,Y)=tell the monitor to replace X, Y with the givens dict, "
                        " but you don't actually want them to be replaced.")
                ipt = X
                if Y is not None:
                    ipt = [X,Y]
                self.monitor.add_channel(prefix+'objective',ipt=ipt,val=obj, dataset=monitoring_dataset, prereqs =  [ prereq ])

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

                    self.monitor.add_channel(name=prefix+name,
                                             ipt=ipt,
                                             val=J, dataset=monitoring_dataset,
                                             prereqs=prereqs)


        self.accumulate = self.combine_batches > 1
        if self.accumulate:
            self.inputs = [elem for elem in [X, Y, drop_mask, drop_mask_Y] if elem is not None]
        else:
            self.inputs = None

        self.optimizer = BatchGradientDescent(
                            objective = obj,
                            inputs = self.inputs,
                            verbose = 1,
                            gradients = gradients,
                            gradient_updates = gradient_updates,
                            params = model.get_params(),
                            lr_scalers = model.get_lr_scalers(),
                            param_constrainers = [ model.censor_updates ],
                            max_iter = self.max_iter,
                            tol = 3e-7,
                            init_alpha = self.init_alpha,
                            reset_alpha = self.reset_alpha,
                            conjugate = self.conjugate,
                            reset_conjugate = self.reset_conjugate,
                            min_init_alpha = self.min_init_alpha,
                            line_search_mode = self.line_search_mode,
                            accumulate = self.accumulate,
                            theano_function_mode = self.theano_function_mode)
        self.X = X


        if self.monitoring_dataset is not None:
            self.monitor.add_channel(name='ave_step_size',
                    ipt=ipt, val = self.optimizer.ave_step_size, dataset=self.monitoring_dataset.values()[0])
            self.monitor.add_channel(name='ave_grad_size',
                    ipt=ipt, val = self.optimizer.ave_grad_size, dataset=self.monitoring_dataset.values()[0])
            self.monitor.add_channel(name='ave_grad_mult',
                    ipt=ipt, val = self.optimizer.ave_grad_mult, dataset=self.monitoring_dataset.values()[0])

        self.first = True
        self.bSetup = True

    def before_step(self, model):
        if self.scale_step != 1.:
            self.params = list(model.get_params())
            self.value = [ param.get_value() for param in self.params ]

    def after_step(self, model):
        if self.scale_step != 1:
            for param, value in safe_zip(self.params, self.value):
                value = (1.-self.scale_step) * value + self.scale_step * param.get_value()
                param.set_value(value)

    def train(self, dataset):
        assert self.bSetup
        model = self.model
        if self.batch_size is None:
            batch_size = model.force_batch_size
        else:
            batch_size = self.batch_size
            if hasattr(model, 'force_batch_size'):
                assert (model.force_batch_size <= 0 or batch_size ==
                        model.force_batch_size)

        assert self.batch_size % self.duplicate == 0
        rng = self.rng
        train_iteration_mode = 'shuffled_sequential'
        if not is_stochastic(train_iteration_mode):
            rng = None
        iterator = dataset.iterator(mode=train_iteration_mode,
                batch_size=self.batch_size // self.duplicate,
                num_batches=self.batches_per_iter,
                targets=self.cost.supervised,
                topo=self.X.ndim != 2,
                rng = rng)

        accum_batches = []

        if self.accumulate:
            warnings.warn("InpaintAlg.train wastes time setting shared variables only to pull their value back out.")

        for data in iterator:
            if self.cost.supervised:
                X, Y = data
                mode = self.theano_function_mode
                if mode is not None and hasattr(mode, 'record'):
                    stry = str(Y).replace('\n',' ')
                    mode.record.handle_line('data Y '+stry+'\n')
                if self.duplicate > 1:
                    Y = np.concatenate([Y] * self.duplicate, axis=0)
                self.Y.set_value(Y)
            else:
                X = data

            if self.duplicate > 1:
                X = np.concatenate([X] * self.duplicate, axis=0)
            self.X.set_value(X)

            self.update_mask()
            if self.accumulate:
                accum_batches.append([elem.get_value() for elem in self.inputs])
                if len(accum_batches) == self.combine_batches:
                    self.before_step(model)
                    self.optimizer.minimize(*accum_batches)
                    self.after_step(model)
                    actual_batch_size = sum([batch[0].shape[0] for batch in accum_batches])
                    model.monitor.report_batch(actual_batch_size)
                    accum_batches = []
            else:
                self.before_step(model)
                self.optimizer.minimize()
                self.after_step(model)
                actual_batch_size = X.shape[0]
                model.monitor.report_batch(actual_batch_size)
        assert len(accum_batches) == 0

    def continue_learning(self, model):
        if self.termination_criterion is not None:
            return self.termination_criterion(self.model)
        return True

class BatchGrower(TrainExtension, TerminationCriterion):

    def __init__(self, channel, available_batches, giveup_after = None,
            reset_best = None):
        """
        Grows the combine_batches setting (to the next factor of available_batches)
        whenever the given channel fails to decrease.

        Gives up when the channel fails to decrease and combine_batches >= giveup_after
        giveup_after defaults to available_batches if unspecified.

        reset_best is a list of epochs after which the best value should be reset, eg,
            because the objective function changes at that point in time.
        """

        if giveup_after is None:
            giveup_after = available_batches
        self.__dict__.update(locals())
        del self.self
        self.continue_learning = True
        self.first = True
        self.best_prev = np.inf
        if reset_best is None:
            self.reset_best = []

    def on_monitor(self, model, dataset, algorithm):
        monitor = model.monitor

        if self.first:
            self.first = False
            assert algorithm.accumulate
            self.monitor_channel = sharedX(algorithm.combine_batches)
            # TODO: make monitor accept channels not associated with any dataset,
            # so this hack won't be necessary
            hack = monitor.channels.values()[0]
            monitor.add_channel('combine_batches', hack.graph_input, self.monitor_channel, dataset=hack.dataset)
        channel = monitor.channels[self.channel]
        v = channel.val_record
        if len(v) == 1:
            return
        latest = v[-1]
        print "Latest "+self.channel+": "+str(latest)
        print "Best previous is "+str(self.best_prev)
        if latest >= self.best_prev:
            cur = algorithm.combine_batches
            print "Looks like using "+str(cur)+" isn't working out so great for us."
            while True:
                cur += 1
                if cur > self.giveup_after:
                    print "Guess we just have to give up."
                    self.continue_learning = False
                    break
                if self.available_batches % cur == 0:
                    print "Let's see how "+str(cur)+" does."
                    algorithm.combine_batches = cur
                    self.monitor_channel.set_value(np.cast[config.floatX](cur))
                    break
        else:
            self.best_prev = latest
        if (len(v) - 1) in self.reset_best:
            print "Resetting our record of the previous best."
            self.best_prev = np.inf


    def __call__(self, model):
        return self.continue_learning


from pylearn2.training_algorithms.bgd import ScaleStep, StepShrinker
