from pylearn2.kmeans import KMeans

k = 400

model = KMeans(k, verbose = True)

import sys

path = sys.argv[1]

import numpy as np

X = np.load(path)

model.train(X)

from pylearn2.utils.serial import save
save('kmeans.pkl', model)
