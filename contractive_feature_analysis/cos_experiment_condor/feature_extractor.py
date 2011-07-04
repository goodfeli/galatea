import numpy as N
from theano import function, shared
import theano.tensor as T

class TanhFeatureExtractor:
    def __init__(self, W, b):
        self.W = shared(W)
        self.b = shared(b)
        self.redo_theano()

    def redo_theano(self):
        X = T.matrix()
        H = T.tanh(T.dot(X,self.W)+self.b)
        self.extract = function([X],H)

        #number examples x number hiddens x number visibles
        J = (1.-T.sqr(H)).dimshuffle(0,1,'x') * self.W.dimshuffle('x',1,0)
        self.jacobian_of_expand = function([X],J)
    #


    @classmethod
    def make_from_examples(X, low, high):
        #for every directed pair of examples (i,j)
        #make a feature that takes on the value low at i
        #and value high at j

        m,n =  X.shape
        h = m **2 - m
        W = N.zeros((n,h))
        idx = 0

        inv_low = N.arctan(low)
        inv_high = N.arctan(high)

        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                if i == j:
                    continue

                diff = X[j,:] - X[i,:]
                direction = diff / N.sqrt(N.square(diff).sum())
                pi = N.dot(X[i,:],direction)
                pj = N.dot(X[j,:],direction)

                wmag =  (inv_high - inv_low) / (pj - pi)

                b[idx] = (pj*inv_low - pi*inv_high) / (pj - pi)
                W[:,idx] = wmag * diff

                #check it
                ival = N.tanh(N.dot(W[:,idx],X[i,:])+b[idx])
                jval = N.tanh(N.dot(W[:,idx],X[j,:])+b[idx])

                assert abs(ival-low) < 1e-6
                assert abs(jval-high) < 1e-6

                idx += 1

        return TanhFeatureExtractor(W,b)
