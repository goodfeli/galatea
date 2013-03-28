import numpy as np
import theano.tensor as T
from theano import function
from theano_linear.matrixmul import MatrixMul
from theano import shared

a = 2
b = 2
c = 2

W1 = shared(np.zeros((a,b), dtype='float32'))
W2 = shared(np.zeros((b,c), dtype='float32'))
transformer = MatrixMul(W2)
b1 = shared(np.zeros((b,), dtype='float32'))

H1 = T.nnet.sigmoid(b1.dimshuffle('x',0)) # bug goes away if I remove the sigmoid
H2 = T.dot(H1, W2)

# theano_linear.matrixmul.MatrixMul is evidently not just a
# a wrapper around T.dot(.,W2.T). The bug goes away if I
# do the matrix multiply the simple way.
msg = transformer.lmul_T(H2)

obj = msg.sum()


grad = T.grad(obj, W2, disconnected_inputs='ignore')
f = function([], grad)

