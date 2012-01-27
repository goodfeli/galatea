import sys
from pylearn2.utils import serial
from theano import shared
in_path = sys.argv[1]

model = serial.load(in_path)

fixedobjs = {}

def fix(obj):
    if hasattr(obj,'__array__'):
        return

    if isinstance(obj, (int, float)):
        return

    fields = dir(obj)
    print 'scanning fields'
    for field in fields:
        print '\t',field
        field_obj = getattr(obj,field)

        if hasattr(field_obj,'__call__'):
            continue

        if field in ['__base__','__class__','__call__','__hash__',
                '__weakref__',
                '__cmp__','__delattr__','__doc__','__getattribute__']:
            continue


        orig_id = id(field_obj)
        print orig_id

        if orig_id in fixedobjs:
            print 'done before'
            setattr(obj,field,fixedobjs[orig_id])
        elif hasattr(field_obj, 'get_value'):
            print 'converting'
            field_obj = shared(field_obj.get_value(borrow=False))
            setattr(obj,field,field_obj)
            fixedobjs[orig_id] = field_obj
        else:
            print 'marking to fix'
            fixedobjs[orig_id] = field_obj
            print 'fixing'
            fix(field_obj)

fix(model)

print 'saving'
serial.save(sys.argv[2],model)
