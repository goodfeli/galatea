from pylearn2.monitor import Monitor
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
import theano.tensor as T
from theano import function
from pylearn2.utils import sharedX
import numpy as np
import warnings

class SetupBatch:
    def __init__(self,alg):
        self.alg = alg

    def __call__(self, * args):
        X = args[0]
        if len(args) > 1:
            warnings.warn('got more arguments than expected')
        self.alg.setup_batch(X)

    def __getstate__(self):
        return {}

class InpaintAlgorithm(object):
    def __init__(self, mask_gen, cost, batch_size=1000, batches_per_iter=10,
                 monitoring_batches=-1, monitoring_dataset=None):
        """
        if batch_size is None, reverts to the force_batch_size field of the
        model
        """
        self.__dict__.update(locals())
        del self.self
        if monitoring_dataset is None:
            assert monitoring_batches == -1
        self.bSetup = False

    def setup_batch(self, X):
        assert not isinstance(X,tuple)
        self.X.set_value(X)
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

        model.cost = self.cost
        model.mask_gen = self.mask_gen

        self.monitor = Monitor.get_monitor(model)
        prereq = self.get_setup_batch_object()
        #We want to use big batches. We need to make several theano calls on each
        #batch. To avoid paying the GPU latency every time, we use a shared variable
        #but the shared variable needs to stay allocated during the time that the
        #monitor is working, and we don't want the monitor to increase the memory
        #overhead. So we make the monitor work off of the same shared variable
        X = sharedX( dataset.get_batch_design(2) , 'X')
        drop_mask = sharedX( np.cast[X.dtype] ( X.get_value() > 0.5 ), 'Y')
        updates = { drop_mask : self.mask_gen(X) }
        self.update_mask = function([], updates = updates)

        obj = self.cost(model,X,drop_mask)
        Y = T.matrix('Y')


        if self.monitoring_dataset is not None:
            if not self.monitoring_dataset.has_targets():
                Y = None
            Y = None
            self.monitor.set_dataset(dataset=self.monitoring_dataset,
                                mode="sequential",
                                batch_size=self.batch_size,
                                num_batches=self.monitoring_batches)
            channels = model.get_monitoring_channels(X,Y)
            if not isinstance(channels, dict):
                raise TypeError("model.get_monitoring_channels must return a "
                                "dictionary, but it returned " + str(channels))
            wtf = self.cost.get_monitoring_channels(model, X = X, drop_mask = drop_mask)
            for key in wtf:
                channels[key] = wtf[key]

            #we only need to put the prereq in once to make sure it gets run
            #adding it more times shouldn't hurt, but be careful
            #each time you say "self.setup_batch" you get a new object with a
            #different id, and if you install n of those the prereq will run n
            #times. It won't cause any wrong results, just a big slowdown
            self.monitor.add_channel('objective',ipt=X,val=obj,prereqs =  [ prereq ])

            for name in channels:
                J = channels[name]
                if isinstance(J, tuple):
                    assert len(J) == 2
                    J, prereqs = J
                else:
                    prereqs = None

                if Y is not None:
                    ipt = (X,Y)
                else:
                    ipt = X

                self.monitor.add_channel(name=name,
                                         ipt=ipt,
                                         val=J,
                                         prereqs=prereqs)


        self.optimizer = BatchGradientDescent(
                            objective = obj,
                            params = model.get_params(),
                            param_constrainers = [ model.censor_updates ],
                            max_iter = 5)
        self.optimizer.verbose = True
        self.X = X


        self.first = True
        self.bSetup = True

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

        for i in xrange(self.batches_per_iter):
            self.X.set_value(dataset.get_batch_design(self.batch_size))
            self.update_mask()
            self.optimizer.minimize()
            model.monitor.report_batch( batch_size )
        return True
