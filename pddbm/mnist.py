from pylearn2.datasets.mnist import MNIST
from pylearn2.utils.image import show

def explore(X,y):
    for i in xrange(X.shape[0]):
        patch = X[i,:].reshape(28,28)
        print y[i]
        show(patch)
        x = raw_input('waiting...')
        if x == 'n':
            return
        if x == 'q':
            quit(-1)

print 'training set'
train = MNIST(which_set = 'train')
explore(train.X,train.y)

print 'test set'
test = MNIST(which_set = 'test')
explore(test.X,test.y)

