import argparse
import time
import gc
import numpy
import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.config import yaml_parse



def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)

        if i % 10 == 0:
            print i, numpy.mean(times), numpy.mean(nlls)

    return numpy.array(nlls)

def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))

def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """

    x = T.matrix()
    mu = theano.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    return theano.function([x], E - Z)


def cross_validate_sigma(samples, data, sigmas, batch_size):

    lls = []
    for sigma in sigmas:
        print sigma
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size = batch_size)
        lls.append(numpy.asarray(tmp).mean())

    ind = numpy.argmax(lls)
    return sigmas[ind]

def main():
    parser = argparse.ArgumentParser(description = 'Parzen window, log-likelihood estimator')
    parser.add_argument('-p', '--path', help='model path')
    parser.add_argument('-s', '--sigma', default = None)
    args = parser.parse_args()

    model = serial.load(args.path)

    src = model.dataset_yaml_src
    batch_size = 100
    num_samples = 10000
    model.set_batch_size(batch_size)

    #assert src.find('train') != -1
    test = yaml_parse.load(src)
    test = test.get_test_set()

    samples = model.generator.sample(num_samples).eval()

    # cross validate simga
    if args.sigma is None:
        rng = numpy.random.RandomState(2014)
        sigma_range = numpy.linspace(0.001, 1., num=2)
        sample_size = 1000
        ind = rng.randint(0, test.X.shape[0], sample_size)
        sigma = cross_validate_sigma(samples[ind], test.X[ind], sigma_range, batch_size)
    else:
        sigma = float(args.sigma)

    print "Using simga: {}".format(sigma)

    gc.collect()

    # fit and evaulate
    parzen = theano_parzen(samples, sigma)
    ll = get_nll(test.X, parzen, batch_size = batch_size)

    print "Log-Likelihood of test set = {}, std: {}".format(ll.mean(), ll.std())

    # valid
    if 0:
        from pylearn2.datasets.mnist import MNIST
        valid = MNIST(which_set='train', start=50000, stop=60000)
        ll = get_nll(valid.X, parzen, batch_size = batch_size)
        print "Log-Likelihood of valid set = {}, std: {}".format(ll.mean(), ll.std())



if __name__ == "__main__":
    main()
