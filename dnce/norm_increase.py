import numpy as np
import matplotlib.pyplot as plt


trials = 100
dim = range(1,1000)
std = .01
y = []

rng = np.random.RandomState([1,2,3])

for cur_dim in dim:
    print cur_dim

    sample_greater = 0
    sample_unchanged = 0
    vertex = rng.randn(cur_dim)
    vertex /= np.sqrt(np.square(vertex).sum())
    norm_sq = np.square(vertex).sum()
    print norm_sq

    for trial in xrange(trials):

        eps = rng.randn(cur_dim) * std
        mask = rng.uniform(0.,1.,(cur_dim,)) < 1./float(cur_dim)
        #print '\t',mask.mean(),1./float(cur_dim),cur_dim
        #assert cur_dim < 6
        if mask.sum() == 0:
            sample_unchanged += 1
        eps *= mask
        sample = vertex + eps
        sample_norm_sq = np.square(sample).sum()

        if sample_norm_sq > norm_sq:
            assert mask.sum() != 0
            sample_greater += 1

    sample_greater = float(sample_greater) / float(trials)
    sample_unchanged = float(sample_unchanged) / float(trials)
    y.append(sample_unchanged+sample_greater)

plt.plot(dim,y)
plt.show()
