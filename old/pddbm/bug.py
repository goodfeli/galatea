import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


rng = np.random.RandomState([1,2,3])

X = rng.randn(1000,1000)
y = rng.randint(0,10,(1000,1))

svm = OneVsRestClassifier(SVC(kernel = 'linear', C = 1.0)).fit(X,y)

print svm.predict(X).shape
