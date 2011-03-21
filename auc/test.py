import numpy as N
import make_learning_curve

def test(model, data, verbose = False):

    #print 'in test'
    #print 'data.X'
    #print data.X.sum()
    #print 'data.Y'
    ##print data.Y.sum()
    #die


    if verbose:
        print '==> Testing hebbian... '
    #

    p, n = data.X.shape

    assert n == model.W.shape[0]

    Yest = N.dot(data.X,model.W.T) + model.b0
    #

    # Remove ties (the negative class is usually most abundant)
    Yest[Yest == 0.] = -1e-12

    #print 'test output'
    #print (data.X.shape, Yest.shape)
    #assert False

    rdata = make_learning_curve.data_struct(Yest, data.Y)

    if verbose:
        print 'done'
    #

    return rdata
