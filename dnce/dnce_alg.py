from pylearn2.monitor import Monitor
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
import theano.tensor as T
from theano import function
from pylearn2.utils import sharedX
from galatea.dnce.dnce import DNCE

class DNCE_Algorithm(object):
    def __init__(self, noise, batch_size=1000, batches_per_iter=10,
                     noise_per_clean = 30,
                 monitoring_batches=-1, monitoring_dataset=None):
        """
        if batch_size is None, reverts to the force_batch_size field of the
        model
        """
        self.batch_size, self.batches_per_iter = batch_size, batches_per_iter
        if monitoring_dataset is None:
            assert monitoring_batches == -1
        self.monitoring_dataset = monitoring_dataset
        self.monitoring_batches = monitoring_batches
        self.bSetup = False
        self.noise = noise
        self.noise_per_clean = noise_per_clean

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

        self.monitor = Monitor.get_monitor(model)
        X = T.matrix()
        Y = T.matrix()
        dnce = DNCE( self.noise)
        if self.monitoring_dataset is not None:
            if not self.monitoring_dataset.has_targets():
                Y = None
            self.monitor.set_dataset(dataset=self.monitoring_dataset,
                                mode="sequential",
                                batch_size=self.batch_size,
                                num_batches=self.monitoring_batches)
            X.tag.test_value = self.monitoring_dataset.get_batch_design(2)
            channels = model.get_monitoring_channels(X,Y)
            if not isinstance(channels, dict):
                raise TypeError("model.get_monitoring_channels must return a "
                                "dictionary, but it returned " + str(channels))

            dnce.noise_per_clean = self.noise_per_clean
            obj = dnce(model,X)
            dnce.noise_per_clean = None
            self.monitor.add_channel('DNCE',ipt=X,val=obj)

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

        X = sharedX( dataset.get_batch_design(1), 'X')
        Y = []
        updates = {}
        for i in xrange(self.noise_per_clean):
            Y_i = sharedX( X.get_value().copy() )
            updates[Y_i] = self.noise.random_design_matrix(X)
            Y.append(Y_i)
        self.update_noise = function([], updates = updates)


        obj = dnce(model,X,Y)

        self.optimizer = BatchGradientDescent(
                            objective = obj,
                            params = model.get_params(),
                            param_constrainers = [ model.censor_updates ],
                            max_iter = 5)
        self.X = X
        self.Y = Y


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
            self.update_noise()
            self.optimizer.minimize()
            model.monitor.report_batch( batch_size )
        return True
