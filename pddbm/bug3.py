import numpy as np
from sklearn.svm import SVC
import time

rng = np.random.RandomState([1,2,3])

m = 1000
n = 1000

X = rng.randn(m,n)
w = rng.randn(n)
b = rng.randn(1)
y = (np.dot(X,w) + b ) > 0

t1 = time.time()
svm = SVC(kernel = 'linear', C = 1.0).fit(X,y)
t2 = time.time()
print 'train time ',t2 - t1


t1 = time.time()
y1 = svm.predict(X)
t2 = time.time()
print 'predict time ',t2 - t1
print '# support vectors:',svm.n_support_
print 'predict time per support vector:',(t2-t1)/float(svm.n_support_.sum())

coef = svm.coef_[0,:]
orig_coef = svm.coef_

t1 = time.time()
f = - np.dot(X, orig_coef.T) + svm.intercept_
y2 = f < 0
print y.shape
print y2.shape
print (y2 == y).shape
quit(-1)
t2 = time.time()
print 'dot product time',t2 -t1

print 'class 1 prevalence ',y.mean()
print 'predict accuracy ',(y1 == y).mean()
print 'dot product accuracy ',(y2 == y).mean()
print 'predict and dot agreement rate',(y1 == y2).mean()

coefs = svm.dual_coef_
assert len(coefs.shape) == 2
assert coefs.shape[0] == 1
coefs = coefs[0,:]
w = np.dot(svm.support_vectors_.T, coefs)

assert np.allclose(w,-coef)

f = np.dot(X,w) + b
y3 = (f < 0)
print 'agreement rate with my method: ',(y3 == y1).mean()

print 'dot prod between sklearn coef_ and my coef_: ',np.dot(w,svm.coef_[0,:])
