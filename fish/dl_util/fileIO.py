#! /usr/bin/env python

import cPickle as pickle
import gzip



def saveToFile(filename, obj, quiet = False):
    ff = gzip.open(filename, 'wb')
    pickle.dump(obj, ff, pickle.HIGHEST_PROTOCOL)
    if not quiet:
        print 'saved to', filename
    ff.close()



def loadFromPklGz(filename):
    with gzip.open(filename, 'rb') as ff:
        ret = pickle.load(ff)
    return ret
