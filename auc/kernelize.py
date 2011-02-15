import pd_check
import numpy as N

def kernelize(object, verbose=False):
    #   replace X by X*X'

    if verbose:
	print '==> Kernelizing...'

    #die

    if 'X' in dir(object):
        object.X = kernelize(object.X)
        object.kernelized = True
    elif 'X0' in dir(object):
        for key in object.X0:
            if key == 'devel':
                continue
            #
            object.X0[key] = kernelize(object.X0[key])
        #
        object.kernelized = True
    else:
        M = N.cast['float64'](object)
        object = N.dot(M,M.T)

        #@data object version of kernelize seems to be what actually got called
        #not kernelize.m
        #and kernelize.m does not do this
        #print 'kernelize calling pd_check'
        #if not pd_check.pd_check(object):
        #    print '\n===> kernelize: warning, matrix data is not pd.'
        #
    #

    return object
#


