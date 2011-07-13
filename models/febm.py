import theano.tensor as T
from theano import function

class FEBM:
    """Free Energy-Based Model: An Energy-Based Model with no latent variables """

    def __init__(self, energy_function):
        self.energy_function = energy_function
        self.batches_seen = 0
        self.examples_seen = 0
    #

    def free_energy(self, X):
        return self.energy_function(X)
    #

    def score(self, X):
        return self.energy_function.score(X)
    #

    def censor_updates(self, updates):
        self.energy_function.censor_updates(updates)
    #

    def get_params(self):
        return self.energy_function.get_params()
    #

    def get_weights(self):
        return self.energy_function.get_weights()
    #

    def get_weights_format(self):
        return self.energy_function.get_weights_format()
    #

    def redo_theano(self):
        X = T.matrix()

        self.E_X_batch_func = function([X],self.energy_function(X))
    #
#
