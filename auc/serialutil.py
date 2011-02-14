import cPickle
import pickle
import os

def load(filepath):
    try:
        f = open(filepath,'rb')
        obj = cPickle.load(f)
        f.close()
        return obj
    except Exception, e:
        #assert False
        exc_str = str(e)
        if len(exc_str) > 0:
            raise Exception("Couldn't open '"+filepath+"' due to: "+str(e))
        else:
            print "Couldn't open '"+filepath+"' and exception has no string. Opening it again outside the try/catch so you can see whatever error it prints on its own."
            f = open(filepath,'rb')
            obj = cPickle.load(f)
            f.close()
            return obj


def save(filepath,obj):
        try:
                file = open(filepath,"wb")
        except Exception, e:
                raise Exception('failed to open '+filepath+' for writing, error is '+str(e))
        ""
        try:
            cPickle.dump(obj,file)
            file.close()
        except Exception, e:
            file.close()
            try:
                file = open(filepath,"wb")
                pickle.dump(obj,file)
                file.close()
            except Exception, e2:
                raise Exception(str(obj)+' could not be written to '+filepath+'by cPickle due to '+str(e)+' nor by pickle due to '+str(e2))
            print 'Warning: '+filepath+' was written by pickle instead of cPickle, due to '+str(e)+' (perhaps your object is eally big?)'


""

def clone_via_serialize(obj):
    str = cPickle.dumps(obj)
    return cPickle.loads(str)

def to_string(obj):
    return cPickle.dumps(obj)

def parent_dir(filepath):
        return '/'.join(filepath.split('/')[:-1])

def mkdir(filepath):
    try:
        os.makedirs(filepath)
    except:
        print "couldn't make directory '"+filepath+"', maybe it already exists"
