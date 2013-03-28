from theano import tensor as T
from theano.sandbox.neighbours import images2neibs

X = T.TensorType(broadcastable = (False,False,False,False), dtype = 'float32')()
Y = images2neibs(X,(2,2))
W = T.matrix()
Z = T.dot(Y,W)
cost = Z.sum()
T.grad(cost,W)
