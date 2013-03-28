n = 2
eps = 1e-6
tol = 1e-3 #stop when variational parameters change by less than this amount in 2 norm

#True distribution:
#   with probability 1-eps, exactly one bit is on. uniform over which is on
#   with probability, eps, some other number of bits is on
#Mean field distribution:
#   All bits are independent
#


# q( h_i ) propto exp( E_[h_-i \sim q] log P(h) )
# Suppose h_i is 0.
# Then we have
# sum_{j \neq i} q_j \Pi_{k \neq i,j} (1-q_k)
# chance of log P(h) being log(1-eps)
# and a one minus that chance of it being eps
# Suppose h_i is 1
# Then we have
# \Pi_{j \neq i} (1-q_j)
# chance of log P(h) being log(1-eps)


import numpy as np
rng = np.random.RandomState([1,2,5])

#q = rng.uniform(0.,1.,(n,))
q = np.zeros(n)
q[0] = 1.

print q

while True:
    prev_q = q.copy()

    order = range(n)
    rng.shuffle(order)

    for var_to_update in order:
        high_prob = 0.

        for i in xrange(n):
            if i == var_to_update:
                continue

            rest_off_prob = 1

            for j in xrange(n):
                if j in [var_to_update, i]:
                    continue
                rest_off_prob *= (1.-q[j])

            high_prob += q[i] * rest_off_prob
        #end for i

        zero_mass = high_prob * np.log(1.-eps) + (1.-high_prob)*np.log(eps)


        high_prob = 1.

        for i in xrange(n):
            if i == var_to_update:
                continue

            high_prob *= (1.-q[i])

        one_mass = high_prob * np.log(1.-eps) + (1.-high_prob)*np.log(eps)

        prob = one_mass / (zero_mass + one_mass)

        q[var_to_update] = prob

    print q

    if np.sqrt(np.sum(np.square(prev_q-q))) < tol:
        break


