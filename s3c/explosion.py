#attempting to create an excitation-based explosion (I don't think it's possible)
hat_h = [1., 1.]

alpha = [.01 , .01]

W = [ -1., 1. ]

beta = 1


w = [ beta * (weight ** 2) for weight in W ]

init_hat_s = [ 1., 1.5 ]

hat_s = [ val for val in init_hat_s ]

#like mu in our current model, except that it isn't gated by h
always_on_mu = [ 0., 0. ]

v = 1

def update():
    rval = []

    for i in xrange(2):
        scaleback = alpha[i] + w[i]
        mean_term = always_on_mu[i]
        data_term = beta * v * W[i]
        j = 1 - i
        interaction_term = - W[i] * W[j] * beta * hat_h[j] * hat_s[j]
        hat_s_i = (mean_term + data_term + interaction_term) / scaleback
        rval.append(hat_s_i)

    return rval


for iter in xrange(100):
    print hat_s

    hat_s = update()
