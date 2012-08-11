from pylearn2.training_callbacks.training_callback import TrainingCallback
from galatea.pddbm.batch_gradient_inference import BatchGradientInference
import numpy as np

class BatchGradientInferenceCallback(TrainingCallback):

    def __init__(self):
        self.tester = None

    def __call__(self, model, dataset, algorithm):

        if self.tester is None:
            self.tester = BatchGradientInference(model)
            self.X, self.Y = dataset.get_batch_design(algorithm.batch_size, include_labels = True)
            if not self.tester.has_labels:
                self.Y = None
            model.kl_fail_log = []

        try:
            results = self.tester.run_inference(self.X, self.Y)

            diff = results['orig_kl'] - results['kl']

            print 'kl failure amount: ',diff
        except AssertionError:
            raise
        except Exception, e:
            print "BatchGradientInferenceCallback failed "
            print e
            diff = np.nan

        model.kl_fail_log.append((model.monitor.get_examples_seen(), diff))


