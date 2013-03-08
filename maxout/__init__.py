import numpy as np

def shuffle(dataset, seed = None):

    if seed is None:
        seed = [2013, 2, 26]

    rng = np.random.RandomState(seed)

    for i in xrange(dataset.X.shape[0]):
        j = rng.randint(dataset.X.shape[0])
        tmp = dataset.X[i,:].copy()
        dataset.X[i,:] = dataset.X[j,:].copy()
        dataset.X[j,:] = tmp
        tmp = dataset.y[i,:].copy()
        dataset.y[i,:] = dataset.y[j,:].copy()
        dataset.y[j,:] = tmp

    return dataset
