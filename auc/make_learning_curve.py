import pd_check
import numpy as N
import kernelize
import os
import math
import train
import test
import auc
import time
from scipy import io

current_path = os.path.abspath(__file__)
rp_path = os.path.dirname(current_path) + "/RP.mat"

def list_idx(mat, row_list, col_list):
    if type(row_list) == type(1):
        row_list = [ row_list ]
    if type(col_list) == type(1):
        col_list = [ col_list ]

    rval = N.zeros( (len(row_list), len(col_list) ) , dtype = mat.dtype )
    for i in xrange(rval.shape[0]):
        for j in xrange(rval.shape[1]):
            rval[i,j] = mat[ row_list[i], col_list[j] ]
        #
    #

    #print 'list_idx input:'
    #print mat
    #print 'list_idx output:'
    #print rval

    return rval
#


class data_struct:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    #

    def subdim(self, pidx = None, fidx = None, lidx = None):
        if pidx is None:
            pidx = range(0, self.X.shape[0] )
        #
        if fidx is None:
            fidx = range(0, self.X.shape[1] )
        #
        if lidx is None:
            lidx = range(0, self.Y.shape[1] )
        #
        return data_struct( self.X[N.ix_(pidx, fidx)], self.Y[N.ix_(pidx,lidx)])
    #
    def copy(self):
        d = data_struct(None, None)
        d.X = self.X.copy()
        d.Y = self.Y.copy()

        return d
#


def randperm(n):
    #print 'warning, randperm replaced'
    return N.random.permutation(n)

    #rval = range(n)
    #for i in xrange(n):
    #    j = N.random.randint(0,n)
    #    temp = rval[i]
    #    rval[i] = rval[j]
    #    rval[j] = temp
    #return rval
#

def make_learning_curve(X, Y, min_repeat, max_repeat, ebar, max_point_num, debug=False, useRPMat=False):


    if debug:
        print "ENTER MLC"

    """x, y, e = make_learning_curve(a, X, Y, min_repeat, max_repeat, ebar, max_point_num)
% Make the learning curve
% Inputs:
% X -- data matrix
% Y -- labels
% Returns:
% x -- Number of samples
% y -- Performance (AUC)
% e -- error bar

"""

    """print 'X'
    print X.shape
    print X.sum()
    print 'Y'
    print Y.shape
    print Y.sum()"""
    #die
    if debug:
        print "Casting X to 64 bit"

    X = N.cast['float64'](X)

    # Verify dimensions and set target values
    p, n = X.shape
    pp, cn = Y.shape
    sep_num=cn

    if pp != p:
        raise Exception('Size mismatch. X has '+str(p)+' examples but Y has '+str(pp)+' labels')
    #

    if cn==2 and N.all(Y.sum(axis=1)): # If only 2 classes, the 2nd col is the same as the first but opposite
        Y=(N.ones((1,1))*Y[:,0]).T
        sep_num=1
    #
    Y[Y==0] = -1;

    if debug:
        time.sleep(2)
        print "Creating the data matrices"

    # Create the data matrices (Y at this stage is still multi-column)
    D = data_struct(X, Y);
    feat_num= D.X.shape[1]

    K = None

    #if not pd_check.pd_check(D.X):
        #die

    #    if debug:
    #        time.sleep(2)
    #        print "kernelizing"
    #    K = kernelize.kernelize(D);
    #
    if not pd_check.pd_check(D.X) and D.X.shape[0] < 2000:
        D = kernelize.kernelize(D)


    #print 'MLC G'

    # Load random splits (these are the same for everyone)
    RP = None
    if useRPMat and os.path.exists(rp_path) and os.path.isfile(rp_path):
        RP = io.loadmat(rp_path, struct_as_record = False)['RP']
        #print RP
        rp, mr = RP.shape

        if rp < p:
            if debug:
                print 'make_learning_curve::warning: RP too small'
            RP = None
        else:
            max_repeat=min(max_repeat, mr)
            RP=N.ceil(N.cast['float64'](RP)/(float(rp)/float(p)))
            RP=RP.astype(int)
            if debug:
                print 'make_learning_curve: using RP of dim '+str(rp)+'x'+str(mr)+' min='+str(RP.min())+' max='+str(RP.max())+', max_repeat='+str(max_repeat)
            #
        #
    else:
        print 'make_learning_curve::warning: no RP file found\n'
    #


    #print 'MLC M'

    if debug:
        time.sleep(2)
        print "Computing sample sizes"

    # Sample sizes scaled in log2
    m = N.floor(math.log(p,2))
    x = 2. ** N.arange(0,int(m)+1)

    if x[-1] != p:
        x = N.hstack((x,[p]))
    #

    x = x[0:-1] # need to remove the last point

    #print 'warning: this is a likely place i could have messed things up'

    if max_point_num is None:
        point_num = x.shape[0]
    else:
        point_num=min(x.shape[0], max_point_num);
    #

    # Loop over the sample sizes
    x = x[0:point_num]
    x = N.cast['uint32'](x)
    y = N.zeros(x.shape)
    e = N.zeros(x.shape)

    for k in xrange(0,point_num):

        if debug:
            print '-------------------- Point %d ----------------------' % k
        #

        A = N.zeros((sep_num,1))
        E = N.zeros((sep_num,1))
        e[k] = N.Inf
        # Loop over number of "1 vs all" separations
        for j in xrange(0,sep_num):

            if debug:
                print ' sep %d -- ' % j
            ""
            repnum = 0
            area = []

            # Loop over repeats (floating number of repeats)
            while repnum < min_repeat or (E[j] > ebar and repnum<max_repeat):

                if debug:
                    #print 'repeat %d **' % repnum
                    #print 'min_repeat: '+str(min_repeat)
                    #print 'max_repeat: '+str(max_repeat)
                    pass
                #

                if RP is None:
                    rp=randperm(p)
                else:
                    rp=RP[0:p, repnum] - 1
                #


                #print 'rp'
                #print rp

                tr_idx = rp[0:x[k]]
                te_idx = rp[x[k]:]

                #print 'te_idx'
                #print len(te_idx)
                #print 'tr_idx'
                #print len(tr_idx)
                #print 'j'
                #print j

                if debug:
                    print "Obtaining sub arrays"
                    time.sleep(2)


                if pd_check.pd_check(D): # kernelized version
                    if debug:
                        print 'pd_check ok, using kernelized version'
                    Dtr = D.subdim(tr_idx, tr_idx, [j])
                    Dte = D.subdim(te_idx, tr_idx, [j])
                #elif x[k] < feat_num: # kernelized too (for speed reason)
                #    if debug:
                #        print 'x[k] < feat_num, using kernelized version'
                #    Dtr = K.subdim(tr_idx, tr_idx, [j]);
                #    Dte = K.subdim(te_idx, tr_idx, [j]);               
                else: # primal version 
                    if debug:
                        print 'using non-kernelized version'
                    Dtr = data_struct(D.X[tr_idx,:], D.Y[tr_idx,j])
                    Dte = data_struct(D.X[te_idx,:], D.Y[te_idx,j])


                #print 'Dte.X'
                #print Dte.X.shape


                if debug:
                    time.sleep(2)
                    print "Training classifier"
                d, m = train.train( Dtr);

                if debug:
                    time.sleep(2)
                    print "Computing test values"
                #print 'Dte.Y'
                #print Dte.Y
                #assert False
                d1 = test.test(m, Dte)
                assert d1.X.shape[0] != 0
                assert repnum == len(area)
                #print 'target'
                #print d1.Y
                #print d1.Y.shape
                #print d1.Y.sum()
                #assert False
                if debug:
                    time.sleep(2)
                    print "Computing auc"
                area.append( auc.auc(d1.X, d1.Y, dosigma=False)[0] )


                if debug:
                    time.sleep(2)
                    print "done"

                repnum += 1
                E[j] = N.asarray(area).std()/N.sqrt(repnum)
            # repnum loop
            assert not N.any(N.isnan(area))
            A[j] = N.asarray(area).mean()
            if N.isnan(A[j]):
                assert False, "Invalid area: " + str(area)
            #
        #end % for j=1:sep_num
        e[k] = E.mean()
        y[k] = A.mean()

        assert not N.isnan(y[k])

        if debug:
            print '==> '+str(repnum)+' repeats, auc='+str(y[k])+'+-'+str(e[k])+' -----------------'

        #
    # % Loop over k

    # Add point with 0 examples
    x = N.concatenate( (N.asarray([0]), x ))
    P = 0.5
    y = N.concatenate( (N.asarray([P]), y) )
    e = N.concatenate( ( N.asarray([N.sqrt(P*(1-P)/p)]), e ) )

    #print "EXIT MLC"

    return x,y,e
#


