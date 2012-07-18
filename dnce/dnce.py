from pylearn2.costs.cost import UnsupervisedCost
import theano.tensor as T
from theano import Variable

class DNCE(UnsupervisedCost):

    def __init__(self, noise_conditional,
            noise_per_clean = None):
        """
            noise_conditional:
                the pylearn2.distributions.Conditional object
                used to corrupt the data
                (currently the data must be corrupted
                outside of this cost so that the same
                corruption can be used repeatedly during
                line searches, but it is still necessary
                to pass the corruptor to the cost so the
                cost can evaluate the probability of the
                noise samples under the corrupting distribution)
            noise_per_clean:
                Optional.
                    If None, then the noisy examples must be passed to
                    __call__ by the caller.
                    Otherwise, it gives an integer number of noise
                    examples that the cost should generate.
        """

        #copy arguments to class body
        self.__dict__.update(locals())
        del self.self

    def __call__(self, model, X, Y = None):
        """
            model: the model to fit to the distribution generating X
            X: a design matrix of examples
            Y: a list of design matrices of corrupted versions of X
                OR None, if noise_per_clean was passed to the constructor
        """

        model.dnce = self

        if self.noise_per_clean is not None:
            assert Y is None
            Y = [ self.noise_conditional.random_design_matrix(X)
                    for i in xrange(self.noise_per_clean) ]
            #This method puts everything in one big BLAS call
            #but forces you to evaluate posoitive energy n times
            #on CPU it is slower, on GPU WAY faster
            #also incompatible with the second method below
            #X = T.concatenate( [ X ] * self.noise_per_clean, axis=0)
            #Y = [ self.noise_conditional.random_design_matrix(X) ]
        else:
            assert Y is not None
            #use the "fast on gpu" version (not compatible with the new criterion)
            #X = T.concatenate( [X ] * len(Y), axis=0)
            #Y = [ T.concatenate( Y , axis = 0) ]

        if 1:
            sub_objs = []

            assert self.noise_conditional.is_symmetric()

            for Y_i in Y:
                assert Y_i.ndim == 2
                #this needs to be negative because as a Cost class we
                #expect this to be minimized
                #sub_obj = -T.log(1./(1.+T.exp(model.free_energy(X)-model.free_energy(Y_i))))
                #sub_obj = -T.log(T.nnet.sigmoid(model.free_energy(Y_i)-model.free_energy(X)))
                sub_obj = T.nnet.softplus(model.free_energy(X)-model.free_energy(Y_i))
                assert sub_obj.ndim == 1
                sub_obj = sub_obj.mean()
                sub_objs.append(sub_obj)

            total_obj = sum(sub_objs) / float(len(sub_objs))

            return total_obj

        if 0: #new objective function Michael proposed at ICML
            sub_objs = []

            assert self.noise_conditional.is_symmetric()

            n = len(Y) - 1

            for Y_i in Y[0:1]:
                #this needs to be negative because as a Cost class we
                #expect this to be minimized
                #sub_obj = -T.log(1./(1.+n*T.exp(model.free_energy(X)-model.free_energy(Y_i))))
                #sub_obj = -T.log(1./(1.+T.exp(T.log(n)+model.free_energy(X)-model.free_energy(Y_i))))
                sub_obj = T.nnet.softplus(T.log(n)+model.free_energy(X)-model.free_energy(Y_i))
                assert sub_obj.ndim == 1
                sub_obj = sub_obj.mean()
                sub_objs.append(sub_obj)

            for Y_i in Y[1:]:
                #this needs to be negative because as a Cost class we
                #expect this to be minimized
                sub_obj = T.nnet.softplus(-T.log(n)+model.free_energy(X)-model.free_energy(Y_i))
                assert sub_obj.ndim == 1
                sub_obj = sub_obj.mean()
                sub_objs.append(sub_obj)

            assert len(sub_objs) == n + 1
            assert len(sub_objs) > 1

            total_obj = sum(sub_objs) / float(len(sub_objs))

            return total_obj
