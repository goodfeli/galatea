import numpy as N
import pd_check
import test

class hebbian_model:
    def __init__(self):
        self.b0 = 0.0

def train( data , verbose = False, do_test = False):
    """ returns data, model 
        # Simple linear classifier with Hebbian-style learning.
        % Inputs:
        % data      -- A data object.
        % Returns:
        % model     -- The trained model.
        % data      -- A new data structure containing the results.
        % Usually works best with standardized data. Standardization is not
        % performed here for computational reasons (we put it outside the CV loop). """


    if verbose:
        print '==> Training Hebbian classifier ... '

    Posidx = N.nonzero(data.Y>0)[0]
    Negidx = N.nonzero(data.Y<0)[0]

    #print data.Y
    #print data.Y > 0
    #print data.Y < 0
    #print Posidx[0]
    #print Posidx[1]
    #print Negidx[0]
    #print Negidx[1]
    #assert False

    model = hebbian_model()

    if pd_check.pd_check(data):
        #Kernelized version
        model.W = N.zeros((data.Y.shape[0],))
        model.W[Posidx] =  1. / (float(Posidx.shape[0])+1e-12);
        model.W[Negidx] = -1. / (float(Negidx.shape[0])+1e-12);
    else:
        n = data.X.shape[1]
        Mu1 = N.zeros((1, n))
        Mu2 = N.zeros((1, n))
        if Posidx.shape[0] > 0:
            Mu1 = data.X[Posidx,:].mean(axis=0)
        #
        if Negidx.shape[0] > 0:
            Mu2 = data.X[Negidx,:].mean(axis=0)
        #
        model.W = Mu1 - Mu2
        B = (Mu1+Mu2)/2.0
        model.b0 = - N.dot(model.W,B.T)
    #

    if do_test:
        rdata = test.test(model,data, verbose=True)
    else:
        rdata = None
    #

    if verbose:
        print 'done'
    #

    return rdata, model
#

