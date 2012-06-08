from galatea.pddbm.batch_gradient_inference import BatchGradientInference
import numpy as np
from pylearn2.utils import sharedX

class BatchGradientInferenceMonitorHack:

    def __init__(self, model):
        self.tester = None
        self.model = model
        self.kl_fail = sharedX(-42.0, 'kl_fail')

    def __call__(self, X, Y = None):

        model = self.model


        #We need to cache the var params so we can restore them later
        #otherwise the monitor will work off of what batch gradient inference
        #finds, not off of what the model found itself
        #TODO: short term improvement: make this a theano function with updates
        #to reduce gpu transfer latency
        #TODO: long term improvement: co-oordinate the efforts of other monitor
        #       channels, so that they run inference, do their thing, then let
        #       this run.
        #       right now we don't know whether this comes before or after inference,
        #       so it has to restore the state when it's done, and it has to
        #       run inference when it starts
        cache = {}
        ip = model.inference_procedure
        obs = ip.hidden_obs
        for key in obs:
            if obs[key] is None:
                continue
            if isinstance(obs[key],list):
                cache[key] = [ elem.get_value() for elem in obs[key] ]
            else:
                cache[key] = ip.hidden_obs[key].get_value()

        if self.tester is None:
            self.tester = BatchGradientInference(model)
            model.kl_fail_log = []

        try:
            results = self.tester.run_inference(X, Y)

            diff = results['orig_kl'] - results['kl']

            #print 'kl failure amount: ',diff
        except AssertionError:
            raise
        except Exception, e:
            print "BatchGradientInferenceCallback failed "
            print e
            diff = np.nan

        #restore state of var params from before we ran batch gradient inference
        for key in obs:
            if obs[key] is None:
                continue
            if isinstance(obs[key],list):
                for elem, val in zip(obs[key], cache[key]):
                    elem.set_value(val)
            else:
                obs[key].set_value(cache[key])

        self.kl_fail.set_value(diff)

