import numpy as np

m = 60000
b = 100

x = np.zeros((m,1),dtype='int32')
rng = np.random.RandomState([1,2,3])

for epoch in xrange(1,51):
    for i in xrange(m/b):
        j = rng.randint(m-b+1)
        x[j:j+b] += 1
    print 'after epoch',epoch

    for i in xrange(x.max()):
        print '\t',i,':',(x==i).sum()

    epoch += 1
