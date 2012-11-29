import theano

X = theano.tensor.alloc(0., 1, 2)
y = X[:,0]
f = theano.function([],y)


