#script to demonstrate that theano leaks memory on the gpu

import numpy as np
from pylearn2.utils import sharedX
from theano import function
import theano
import gc
import sys

s = [400,8000]
print 'first shared'
#before =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
W = sharedX(np.zeros((s[0],s[1])))
gc.collect()
gc.collect()
gc.collect()
#after =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
#diff = before[0] - after[0]
#expected_diff = s[0]*s[1]*4

#if diff > expected_diff:
#    print "W uses ",str(float(diff)/float(expected_diff))," times more memory than needed."
#    print "(",str(float(diff-expected_diff)/(1024. ** 2))," megabytes)"

print 'second shared'
grad  =sharedX(np.zeros(W.get_value().shape))

init_array = grad.get_value(borrow=True, return_internal_type = True)
print 'initial cudandarray addr %x' % id(init_array)

gc.collect()
gc.collect()
gc.collect()
#after_after = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
#diff = after_after[0] - after[0]

#if diff > expected_diff:
#    print "grad uses ",str(float(diff)/float(expected_diff))," times more memory than needed."


updates = { grad : W}


f = function([], updates = updates)


from theano.printing import debugprint
debugprint(f)

print 'call'

#before =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
f()
gc.collect(); gc.collect(); gc.collect()
#after = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()

print 'references to initial array: ',sys.getrefcount(init_array)

print "ALL DEALLOCS AFTER HERE ARE TOO LATE"


print '--------------------------------'

culprits = gc.get_referrers(init_array)

for culprit in culprits:
    if culprit is locals():
        print '\t<locals()>'
    else:
        print '\t',culprit

print '--------------------------------'


skip = {
        '__call__' : 'would cause infinite loop (property?)',
        '__class__' : 'would cause infinite loop (property?)',
        '__cmp__' : 'would cause infinite loop (property?)',
        '__reduce__' : 'would cause infinite loop (property?)',
        '__reduce_ex__' : 'would cause infinite loop (property?)',
        '__doc__' : 'would cause infinite loop (property?)',
        '__sizeof__' : 'would cause infinite loop (property?)',
        '__str__' : 'would cause infinite loop (property?)',
        '__repr__' : 'would cause infinite loop (property?)',
        '__subclasshook__' : 'would cause infinite loop (property?)',
        '__ge__' : 'would cause infinite loop (property?)',
        '__new__' : 'would cause infinite loop (property?)',
        '__formatter__' : 'would cause infinite loop (property?)',
        '__name__' : 'would cause infinite loop (property?)',
        '__get_item__' : 'would cause infinite loop (property?)',
        '__contains__' : 'would cause infinite loop (property?)',
        '__find__' : 'would cause infinite loop (property?)',
        '__get_name__' : 'would cause infinite loop (property?)',
        '__setattr__' : 'would cause infinite loop (property?)',
        '__func__' : 'would cause infinite loop (property?)',
        '__closure__' : 'would cause infinite loop (property?)',
        '__add__' : 'would cause infinite loop (property?)',
        '__eq__' : 'would cause infinite loop (property?)',
        '__delattr__' : 'would cause infinite loop (property?)',
        '__hash__' : 'would cause infinite loop (property?)',
        '__init__' : 'would cause infinite loop (property?)',
        '__getattribute__' : 'would cause infinite loop (property?)',
        'T' : 'would cause infinite loop (property?)',
        'imag' : 'would cause infinite loop (property?)',
        '__format__' : 'would cause infinite loop (property?)',
        'denominator' : 'wcil (p?)'
}

searched_objs = set([])

found = []

def find(obj, name):
    global searched_objs
    global skip
    global found

    if id(obj) in searched_objs:
        return

    #print len(searched_objs)
    print name

    searched_objs = searched_objs.union([id(obj)])

    if obj is init_array:
        found.append(name)
    for field_name in dir(obj):
        try:
            if field_name in skip.keys():
                reason = skip[field_name]
                #print 'skipping '+name+'.'+field_name+reason
                continue
            field_obj = getattr(obj,field_name)
            find(field_obj,name+'.'+field_name)
        except:
            print name+'.'+field_name+' caused an error'
    if hasattr(obj,'__iter__'):
        for i, subobj in enumerate(obj):
            find(subobj,name + '[' + str(i) + ']')
    if isinstance(obj,dict):
        for key in obj:
            find(obj[key],name+'['+str(key)+']')


#find(locals(),'locals')

print 'found instances: ',found

final_array = grad.get_value(borrow=True, return_internal_type = True)
print 'final cudandarray addr %x' % id(final_array)
#addr = final_array.gpudata
#print 'final storage address: %x' % addr

"""
print '------------'
for item in f.defaults:
    print item
print '------------'
for field in dir(f.defaults):
    print field
print '-----------'
"""

print '-------------'
print f.defaults
print type(f.defaults)
print '--------------'

for field in ['defaults','__dict__']:
    if field in ['container','__weakref__',
            '__subclasshook__','__str__','__sizeof__','__setitem__', \
            '__setattr__','__repr__','__reduce_ex__','__reduce__','__new__','__module__', \
            '__init__','__hash__','__getitem__','__getattribute__','__call__', \
            '__class__','__contains__','__copy__','__delattr__',  '__doc__', '__format__']:
        continue
    print 'deleting f.'+field
    delattr(f,field)
    gc.collect()
    gc.collect()
    gc.collect()
    print 'references to initial array: ',sys.getrefcount(init_array)


print 'deleting f'
del f
gc.collect()
gc.collect()
gc.collect()
print 'references to initial array: ',sys.getrefcount(init_array)


print 'deleting theano'
del theano
gc.collect()
gc.collect()
gc.collect()
print 'references to initial array: ',sys.getrefcount(init_array)


print 'deleting a bunch of stuff'
del skip
del find
del culprits
#del gc
del final_array
#del diff
#del before
del sharedX
#del addr
#del __package__
del np
#del __doc__
del function
#__builtins__', '__file__'
#del after
#sys', '
del updates
del W
del found
#', '__name__', '
del debugprint
del culprit
del s
#del after_after
#del expected_diff
del grad
del searched_objs

gc.collect()
gc.collect()
gc.collect()
print 'references to initial array: ',sys.getrefcount(init_array)


print locals().keys()

#assert after[0] >= before[0]
