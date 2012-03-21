from pylearn2.training_callbacks.training_callback import TrainingCallback
from galatea.pddbm.batch_gradient_inference import BatchGradientInference

class BatchGradientInferenceCallback(TrainingCallback):

    def __init__(self):
        self.tester = None

    def __call__(self, model, dataset, algorithm):

        if self.tester is None:
            self.tester = BatchGradientInference(model)
            self.X = dataset.get_batch_design(100)
            model.kl_fail_log = []

        results = self.tester(self.X)

        diff = results['orig_kl'] - results['kl']

        print 'kl failure amount: ',diff

        model.kl_fail_log.append((model.monitor.examples_seen, diff))


