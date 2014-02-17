#! /usr/bin/env python

'''
Research code

Jason Yosinski
'''

import ipdb as pdb
import hashlib
import marshal
import os
from datetime import datetime
import time
from numpy import *
import cPickle as pickle
import types
import inspect

from fileIO import loadFromPklGz, saveToFile
from misc import mkdir_p



# Directory to use for caching
#globalCacheDir     = '/tmp/pycache'
globalCacheDir     = os.path.join(os.path.expanduser("~"), '.pycache')    # symlink to local disk if desired

globalCacheVerbose = 2                  # 0: print nothing. 1: Print info about hits or misses. 2: print filenames. 3: print hash steps
globalDisableCache = False              # Set to True to disable all caching



__all__ = ['globalCacheDir', 'globalCacheVerbose', 'globalDisableCache', 'memoize', 'cached', 'betacached']



class PersistentHasher(object):
    '''Hashes, persistently and consistenly. Suports only two methods:
    update and hexdigest. Supports numpy arrays and dicts.'''

    def __init__(self, verbose = None):
        self.verbose = verbose if verbose is not None else globalCacheVerbose
        self.counter = 0
        self.hashAlg = hashlib.sha1()
        if self.verbose >= 3:
            self._printStatus()
        self.salt = '3.14159265358979323'
        self.classHashWarningPrinted = False


    def update(self, obj, level = 0):
        '''A smarter, more persistent verison of hashlib.update'''
        if isinstance(obj, types.BuiltinFunctionType):
            # Function name is sufficient for builtin functions.
            # Built in functions do not have func_code, but their code
            # is unlikely to change anyway.
            self.hashAlg.update(obj.__name__)
        elif isinstance(obj, types.FunctionType):
            # For user defined functions, hash the function name and
            # function code itself (a bit overconservative)
            self.hashAlg.update(obj.__name__)
            self.hashAlg.update(marshal.dumps(obj.func_code))
        elif type(obj) is ndarray:
            # can update directly with numpy arrays
            self.hashAlg.update(self.salt + 'numpy.ndarray')
            self.hashAlg.update(obj)
        elif type(obj) is dict:
            self.hashAlg.update(self.salt + 'dict')
            for key,val in sorted(obj.items()):
                self.hashAlg.update(str(hash(key)))
                self.update(val, level = level + 1)  # recursive call
        elif inspect.isclass(obj):
            if not self.classHashWarningPrinted:
                print 'WARNING: hashing classes in util.cache is experimental, proceed with caution!'
                self.classHashWarningPrinted = True
            #raise Exception('Hashing whole classes not supported yet (have not implemented reliable way to hash all methods)')
            self.hashAlg.update(self.salt + 'object')
            for key,val in sorted(obj.__dict__.items()):
                self.hashAlg.update(str(hash(key)))
                self.update(val, level = level + 1)  # recursive call
        else:
            # Just try to hash it
            try:
                self.hashAlg.update(str(hash(obj)))
                #print '  updating with obj hash', str(hash(obj))
            except TypeError:
                if type(obj) is tuple or type(obj) is list:
                    # Tuples are only hashable if all their components are.
                    self.hashAlg.update(self.salt + ('tuple' if type(obj) is tuple else 'list'))
                    for item in obj:
                        self.update(item, level = level + 1)  # recursive call
                else:
                    print 'UNSUPPORTED TYPE: FIX THIS!'
                    print type(obj)
                    print obj
                    print 'UNSUPPORTED TYPE: FIX THIS!'
                    pdb.set_trace()

        self.counter += 1
        if self.verbose >= 3:
            self._printStatus(level, repr(type(obj)))


        # Maybe useful for some other types:
        #        elif type(arg) is dict:
        #            for k,v in sorted(arg.items()):
        #                argsHash.update(k)
        #                argsHash.update(pickle.dumps(v, -1))


    def hexdigest(self):
        return self.hashAlg.hexdigest()


    def _printStatus(self, level = 0, typeStr = None):
        st = '%sAfter %3d objects hashed, hash is %s' % ('    ' * level, self.counter, self.hexdigest()[:4])
        if typeStr is not None:
            st += ' (latest %s)' % typeStr
        print st
        #if self.hexdigest()[:4] == '0211':
        #    pdb.set_trace()



def persistentHash(object):
    '''Convenience function to return hash of single object'''
    ph = PersistentHasher()
    ph.update(object)
    return ph.hexdigest()



def memoize(function):
    '''Decorator to memoize function'''

    if globalDisableCache:

        def wrapper(*args, **kwargs):
            if globalCacheVerbose >= 1:
                print ' -> cache.py: cache disabled by globalDisableCache'
            return function(*args, **kwargs)

    else:

        def wrapper(*args, **kwargs):
            startHashWall = time.time()

            # Hash the function, its args, and its kwargs
            hasher = PersistentHasher()
            hasher.update(function)
            hasher.update(args)
            hasher.update(kwargs)

            # Check cache for previous result
            functionName = function.__name__    # a little more reliable than func_name
            digest = hasher.hexdigest()

            cacheFilename    = '%s.%s.pkl.gz' % (digest[:16], functionName)
            # get a unique filename that does not affect any random number generators
            cacheTmpFilename = '.%s-%06d.tmp' % (cacheFilename, datetime.now().microsecond)
            cachePath    = os.path.join(globalCacheDir, cacheFilename[:2], cacheFilename)
            cacheTmpPath = os.path.join(globalCacheDir, cacheFilename[:2], cacheTmpFilename)
            elapsedHashWall = time.time() - startHashWall

            try:
                start = time.time()
                if globalCacheVerbose >= 3:
                    print (' -> cache.py: %s: trying to load file %s'
                           % (functionName, cachePath))
                (stats,result) = loadFromPklGz(cachePath)
                elapsedWall = time.time() - start
                if globalCacheVerbose >= 1:
                    print (' -> cache.py: %s: cache hit (%.04fs hash overhead, %.04fs to load, saved %.04fs)'
                           % (functionName, elapsedHashWall, elapsedWall, stats['timeWall'] - elapsedWall))
                    if globalCacheVerbose >= 2:
                        print '   -> loaded %s' % cachePath
            except IOError:
                if globalCacheVerbose >= 3:
                    print (' -> cache.py: %s: cache miss, computing function'
                           % (functionName))
                startWall = time.time()
                startCPU  = time.clock()
                result = function(*args, **kwargs)
                elapsedWall = time.time() - startWall
                elapsedCPU  = time.clock() - startCPU
                    
                stats = {'functionName': functionName,
                         'timeWall': elapsedWall,
                         'timeCPU': elapsedCPU,
                         'saveDate': datetime.now(),
                         }

                startSave = time.time()
                mkdir_p(os.path.dirname(cachePath))
                if globalCacheVerbose >= 3:
                    print (' -> cache.py: %s: function execution finished, saving result to file %s'
                           % (functionName, cachePath))
                saveToFile(cacheTmpPath, (stats,result), quiet = True)
                os.rename(cacheTmpPath, cachePath)
                if globalCacheVerbose >= 1:
                    print (' -> cache.py: %s: cache miss (%.04fs hash overhead, %.04fs to save, %.04fs to compute)'
                           % (functionName, elapsedHashWall, time.time() - startSave, elapsedWall))
                    if globalCacheVerbose >= 2:
                        print '   -> saved to %s' % cachePath

            return result

    return wrapper



def cached(function, *args, **kwargs):
    '''Return cached answer or compute and cache.'''

    memoizedFunction = memoize(function)
    return memoizedFunction(*args, **kwargs)



def cached2(cacheobj, function, *args, **kwargs):
    '''Return cached answer or compute and cache. Uses two layers of caching (local object for use with ipython %run and standard caching).'''

    return _cached2(cacheobj, True, function, *args, **kwargs)



def cached2jm(cacheobj, function, *args, **kwargs):
    '''Return cached answer or compute and cache. Uses "Just the Memory" layer of caching (no on-disk caching)'''

    return _cached2(cacheobj, False, function, *args, **kwargs)



def _cached2(cacheobj, cacheFallback, function, *args, **kwargs):
    '''Return cached answer or compute and cache. Uses two layers of caching (local object for use with ipython %run and standard caching).'''

    # Hash the function, its args, and its kwargs
    hasher = PersistentHasher()
    hasher.update(function)
    hasher.update(args)
    hasher.update(kwargs)

    # Check cache for previous result
    functionName = function.__name__    # a little more reliable than func_name
    digest = hasher.hexdigest()

    cacheKey = '%s:%s' % (functionName, digest[:16])

    try:
        ret = cacheobj[cacheKey]
        #print 'hit'
    except KeyError:
        #print 'miss'
        if cacheFallback:
            ret = cached(function, *args, **kwargs)
        else:
            ret = function(*args, **kwargs)
        cacheobj[cacheKey] = ret
    return ret



def invNoncached(mat, times = 1.0):
    return times * linalg.inv(mat)



@memoize
def invCached(mat, times = 1.0):
    return times * linalg.inv(mat)



def main():
    random.seed(0)
    a = random.rand(500,500)
    #print 'a is\n', a
    
    #ainv = invNoncached(a)
    #ainv = invCached(a)
    #ainv = linalg.inv(a)

    print 'computing ainv * 1.0'
    invCached(a, times = 1.0)
    print 'computing ainv * 2.0'
    invCached(a, 2.0)
    print 'computing cached(linalg.inv, a)'
    cached(linalg.inv, a)



if __name__ == '__main__':
    main()
