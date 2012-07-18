import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState([1,2,3])

noise_beta = .1
noise_std = 1/np.sqrt(noise_beta)

trials = 10000

x = []
y = []
z = []

for n in xrange(2,100):
    print n

    results = []
    model_results = []

    for t in xrange(trials):
        samples = rng.randn(n)
        samples[1:] *= noise_std
        samples[1:] += samples[0]


        ev = np.zeros((n,))

        for i in xrange(n):
            ev[i] = np.exp( -0.5 * noise_beta * np.square( samples- samples[i]).sum() )
        ev /= ev.max()

        results.append(ev[0] / ev.sum() )

        prior = np.exp( -0.5 * np.square(samples) )
        ev *= prior
        model_results.append( ev[0] / ev.sum() )

    x.append(n)
    #chance of picking right point
    #y.append( sum(results) / float(len(results) ) )
    #ratio of accuracy versus chance
    y.append( n * sum(results) / float(len(results)) )
    #z.append( sum(model_results) / float(len(model_results)) )
    z.append( n * sum(model_results) / float(len(model_results)) )

plt.plot(x,y)
plt.hold(True)
plt.plot(x,z)
plt.show()
