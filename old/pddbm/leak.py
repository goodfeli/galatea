from theano import shared
import numpy as np
import sys

shape = (1,)

val = np.zeros(shape)
print sys.getrefcount(val)
x = shared(val, borrow=True)
print sys.getrefcount(val)
del x
del shared
del np
del __builtins__
del __package__
del __doc__
del __name__
del __file__
del shape
del sys

print globals().keys()
print locals().keys()


import gc
for i in xrange(100):
    gc.collect()
import sys

print sys.getrefcount(val)
