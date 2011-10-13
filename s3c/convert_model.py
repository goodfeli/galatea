import sys
from pylearn2.utils import serial
from theano import shared
in_path = sys.argv[1]

model = serial.load(in_path)

fixedobjs = {}

def fix(obj):
    if hasattr(obj,'__array__'):
        return

    for field in dir(obj):
        field_obj = getattr(obj,field)

        if hasattr(field_obj,'__call__'):
            continue

        if field in ['__base__','__class__','__call__','__hash__',
                '__weakref__',
                '__cmp__','__delattr__','__doc__','__getattribute__']:
            continue

        print field

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
            print 'fixing'
            fixedobjs[orig_id] = field_obj
            fix(field_obj)

fix(model)

print 'saving'
serial.save(sys.argv[2],model)
