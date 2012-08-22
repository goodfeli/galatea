from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import sharedX
import theano.tensor as T
from theano import function

dataset = MNIST(which_set = 'train', one_hot = False, shuffle = True)
X = dataset.X
y = dataset.y
assert y.ndim == 1

mask = (y == 7) + (y == 9)
X = X[ mask,:]
y = y[mask]
y = y == 7
y = y.astype('float32')

dataset = MNIST(which_set = 'test', one_hot = False)
ty = dataset.y
mask = (ty == 7) + (ty == 9)
tX = dataset.X[mask,:]
ty = ty[mask]
ty = ty == 7
ty = ty.astype('float32')

tX -= X.mean(axis=0)
X -= X.mean(axis=0)
tX /= .01 + X.std(axis=0)
X /= .01 + X.std(axis=0)
#steal upper left pixel as bias term
X[:,0] = 1.
tX[:,0] = 1.


#from pylearn2.gui.patch_viewer import make_viewer
#viewer = make_viewer(X[0:100,:], activation = y[0:100])
#viewer.show()

import numpy as np
rng = np.random.RandomState([1,2,3])

init_w = .01 * rng.randn(784)

batch_size = 100

class Logreg:
    def __init__(self, init_w):
        self.w = sharedX(init_w)
        self.b = sharedX(0.)

        params = [self.w ]

        X = T.matrix()
        y = T.vector()

        X.tag.test_value = np.zeros((100,784),dtype='float32')
        y.tag.test_value = np.zeros((100,),dtype='float32')

        self.cost = function([X,y],self.cost_samples(X,y).mean())
        alpha = T.scalar()
        alpha.tag.test_value = 1.

        cost_samples = self.cost_samples(X,y)
        assert cost_samples.ndim == 1

        cost = cost_samples.mean()
        assert cost.ndim == 0

        updates = {}

        for param in params:
            updates[param] = param - alpha * T.grad(cost,param)

        self.sgd_step = function([X,y,alpha],updates = updates)

        num_samples = cost_samples.shape[0]
        cost_variance = T.sqr(cost_samples-cost).sum() / ( num_samples - 1)
        cost_std = T.sqrt(cost_variance)
        assert cost_std.ndim == 0

        caution = -2.

        bound = cost + caution * cost_std / T.sqrt(num_samples)

        updates = {}

        for param in params:
            updates[param] = param - alpha * T.grad(cost,param)

        self.do_step = function([X,y,alpha],updates = updates)
        self.experimental_step = function([X,y,alpha],updates = { self.w: self.w - alpha * T.grad(bound,param) } )

        alphas = T.vector()
        alphas.tag.test_value = np.ones((2,),dtype='float32')

        #also tried using grad of bound instead of cost (got to change it in do_step as well)
        W = self.w.dimshuffle(0,'x') - T.grad(cost,self.w).dimshuffle(0,'x')* alphas.dimshuffle('x',0)
        B = self.b.dimshuffle('x') - T.grad(cost, self.b).dimshuffle('x') * alphas

        Z = T.dot(X,W) + B
        C = y.dimshuffle(0,'x') * T.nnet.softplus(-Z) + (1-y.dimshuffle(0,'x'))*T.nnet.softplus(Z)

        means = C.mean(axis=0)
        variances = T.sqr(C-means).sum(axis=0) / (num_samples - 1)
        stds = T.sqrt(variances)
        bounds = means + caution * stds / T.sqrt(num_samples)

        self.eval_bounds = function([X,y,alphas],bounds)


        W = T.concatenate( [self.w.dimshuffle('x',0) ] * batch_size, axis= 0)

        z = (X*W).sum(axis=1)

        C = y*T.nnet.softplus(-z) + (1-y)*T.nnet.softplus(z)

        grad_W = T.grad(C.sum(),W)

        zero_mean = grad_W - grad_W.mean()

        cov = T.dot(zero_mean.T,zero_mean)

        from theano.sandbox.linalg import matrix_inverse

        inv = matrix_inverse(cov + np.identity(784).astype('float32') * .01)

        self.nat_grad_step = function([X,y,alpha], updates = { self.w : self.w - alpha * T.dot( inv, T.grad(cost,self.w)) } )




    def cost_samples(self, X, y):
        z = T.dot(X,self.w)+self.b
        return y*T.nnet.softplus(-z)+(1-y)* T.nnet.softplus(z)

    #def experimental_step(self, X,y):
    #    alphas = np.asarray([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1], dtype = 'float32')
    #    bounds = self.eval_bounds(X,y,alphas)
    #    best = self.do_step(X,y,alphas[np.argmax(bounds)])




sgd_model = Logreg(init_w)
experimental = Logreg(init_w)

alpha = 1e-3
print '\tinit costs: ',sgd_model.cost(tX,ty),experimental.cost(tX,ty)
i = 0
while True:
    idx = rng.randint(0,X.shape[0]-batch_size)
    batch_X = X[idx:idx+batch_size,:]
    batch_y = y[idx:idx+batch_size]
    sgd_model.sgd_step(batch_X,batch_y,alpha)
    experimental.experimental_step(batch_X,batch_y,alpha)#nat_grad_step(batch_X,batch_y, alpha)
    if i % 1000 == 0:
        print 'step ',i
        print '\tcosts: ',sgd_model.cost(tX,ty),experimental.cost(tX,ty)
    i = i + 1








