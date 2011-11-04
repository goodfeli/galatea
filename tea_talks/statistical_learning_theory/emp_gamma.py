phi_inc = .01
m = 50000
trials = 1000
cutoff = 950

import numpy as np
rng = np.random.RandomState([1,2,3])

phi = 0.0

phis = []
gammas = []

while phi <= 1.0:
    print 'evaluating phi ',phi

    x = rng.uniform(0,1,(m,trials)) < phi
    phi_hats = x.mean(axis=0)

    errs = abs(phi_hats - phi)

    sorted_errs = sorted(errs)

    gamma = sorted_errs[cutoff]


    gammas.append(gamma)
    phis.append(phi)

    phi += phi_inc


import matplotlib.pyplot as plt
plt.title('Empirical generalization bounds for $m=50000, \delta=.05$')
plt.ylabel('Empirical $\gamma$')
plt.xlabel('$\phi$')
plt.plot(phis,gammas)
plt.show()

