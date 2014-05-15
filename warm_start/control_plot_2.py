num_points = 100
dim = 1e6

import numpy as np

rng = np.random.RandomState([1, 2, 3])

traj = [rng.randn(dim)]
for i in xrange(num_points - 1):
    traj.append(traj[-1] + rng.randn(dim))

traj = np.array(traj)

traj -= traj[0,:]

d1 = traj[-1, :].copy()

d1 /= np.sqrt(np.square(d1).sum())

x = np.dot(traj, d1)

proj = np.outer(x, d1)

traj -= proj

norms = np.sqrt(np.square(traj).sum(axis=1))

i = np.argmax(norms)

d2 = traj[i,:].copy()

d2 /= np.sqrt(np.square(d2).sum())

y = np.dot(traj, d2)

proj = np.outer(y, d2)

traj -= proj


norms = np.sqrt(np.square(traj).sum(axis=1))


from matplotlib import pyplot

pyplot.plot(x, norms, marker='x')
pyplot.hold(True)
pyplot.plot(x, y, marker='o')

pyplot.show()
