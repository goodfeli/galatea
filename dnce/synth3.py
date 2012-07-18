#An experiment with synthetic data to test whether DNCE works.
#For the data-dependent noise model we use additive gaussian noise
#The data is just samples from a zero mean, unit precision univariate
#gaussian. We initialize the model with the wrong precision and see
#how close we can come to recovering the correct precision, and which
#noise precisions are the best.

#Imports
from matplotlib import pyplot as plt
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent
from pylearn2.models.mnd import DiagonalMND
from pylearn2.models.mnd import kl_divergence
from pylearn2.distributions.mnd import MND
from pylearn2.distributions.mnd import AdditiveDiagonalMND
import numpy as np
from pylearn2.utils import sharedX
from theano import function
from galatea.dnce.dnce import DNCE
import theano.tensor as T

#====Options controlling the experiment=========
#the dimension of the data
dim = 1
#number of training examples
m = 20
#number of noise examples per training example
noise_per_clean = 30
#the parameters of the data distribution
true_mu = 1.
true_beta = 1.
#for each of the noise components, we try
#num_beta different values of beta, spaced
#uniformly in log space from 10^min_exp
#to 10^max_exp
num_beta = 20
min_exp = -3.
max_exp = 2.
#number of trials to run
trials = 10


#Generate the values of beta to consider
idxs = np.arange(num_beta)
pos = idxs / float(num_beta-1)
scaled_shifted = pos * (max_exp-min_exp) + min_exp
betas = 10 ** scaled_shifted


kls = np.zeros((trials,num_beta))
ml_kls = np.zeros((trials,))

for trial in xrange(trials):
#generate the data
    data_distribution = MND( sigma = np.identity(dim) / true_beta,
                            mu = np.zeros((dim,)), seed = 17 * (trial+1) )
    true = DiagonalMND( nvis = dim, init_beta = true_beta, init_mu = 0.,
            min_beta = .1, max_beta = 10.)
    X = sharedX(function([],data_distribution.random_design_matrix(m))())

    Xv = X.get_value()
    mu = Xv.mean(axis=0)
    print 'maximum likelihood mu: ',mu
    diff = Xv - mu
    var = np.square(diff).mean(axis=0)
    mlbeta = 1./var
    print 'maximum likelihood beta: ',mlbeta
    ml_model = DiagonalMND( nvis = dim, init_mu = mu, init_beta = mlbeta,
            min_beta = 0.0,
            max_beta = 1e6)
    ml_kl = kl_divergence( true, ml_model)
    ml_kl = function([],ml_kl)()
    assert ml_kl >= 0.0
    ml_kls[trial] = ml_kl
    print 'maximum likelihood kl divergence:',ml_kl

    best_mse = None

    #Try each noise beta
    for idx1 in xrange(num_beta):
        beta = betas[idx1]

        print 'Running experiment for ',beta

        #Allocate a fresh model
        model = DiagonalMND(
                nvis = dim,
                init_mu = 0.,
                init_beta = .1,
                min_beta = .001,
                max_beta = 1e30)

        #Make the noise distribution
        noise_distribution = AdditiveDiagonalMND(
                                    init_beta = beta,
                                    nvis = dim
                                    )

        #generate the noise samples
        noise_func = function([], noise_distribution.random_design_matrix(X))
        Y = []
        for i in xrange(noise_per_clean):
            Y.append(sharedX(noise_func()))

        #Get the objective function
        nce = DNCE(noise_distribution)
        J = nce(model,X,Y)

        accs = []
        for Y_i in Y:
            pos_prob = 1./(1.+T.exp(model.free_energy(X)-model.free_energy(Y_i)))
            acc = (pos_prob > .5).mean()
            accs.append(acc)
        acc = sum(accs) / float(len(accs))

        print '\tinit accuracy ',function([],acc)()

        #Minimize the objective function with batch gradient descent
        minimizer = BatchGradientDescent( objective = J,
                                            params = model.get_params(),
                                            param_constrainers = [ model.censor_updates ])

        print '\tinit obj:',minimizer.obj()
        #minimizer.verbose = True
        minimizer.minimize()
        print '\tfinal obj:',minimizer.obj()

        recovered_beta = model.beta.get_value()
        recovered_mu = model.mu.get_value()

        print '\trecovered beta:',recovered_beta
        print '\trecovered mu:',recovered_mu

        kl = kl_divergence(true, model)
        kl = function([],kl)()
        assert kl >= 0.0

        print '\tkl was ',kl
        print '\tfinal accuracy ',function([],acc)()
        kls[trial,idx1] = kl

plt.hold(True)
plt.plot(betas, kls.mean(axis=0),'b')
plt.plot(betas, kls.mean(axis=0)+kls.std(axis=0),'b--')
plt.plot(betas, kls.mean(axis=0)-kls.std(axis=0),'b--')
plt.plot(betas, ml_kls.mean() *np.ones((num_beta,)),'g')
plt.plot(betas, (ml_kls.mean()+ml_kls.std()) *np.ones((num_beta,)),'g--')
plt.plot(betas, (ml_kls.mean()-ml_kls.std()) *np.ones((num_beta,)),'g--')
plt.ylabel('KL divergence')
plt.xlabel('Noise precision')
ax = plt.gca()
ax.set_xscale('log')
plt.show()
