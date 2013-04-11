import numpy

import theano
import theano.tensor as TT


def safe_clone(cost, old_vars, new_vars):
    dummy_params = [x.type() for x in old_vars]
    dummy_cost = theano.clone(cost,
                              replace=zip(old_vars, dummy_params))
    return theano.clone(dummy_cost,
                        replace=zip(dummy_params, new_vars))


class forloop(theano.gof.Op):
    def __init__(self, loc_fn, n_steps, args, outputs):
        self.loc_fn = loc_fn
        self.n_steps = n_steps
        self.inputs = args
        self.outputs = outputs
        self.reset = theano.function([], [],
            updates=[(x, TT.zeros_like(x)) for x in self.outputs])

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, *args):
        return theano.gof.Apply(self, args, [x.type() for x in args])

    def perform(self, node, inputs, outputs):
        for out in self.outputs:
            out.container.storage[0][:] = 0

        for inp, inp_var in zip(inputs, self.inputs):
            inp_var.set_value(inp, borrow=False)

        for step in xrange(self.n_steps):
            self.loc_fn(step)
        for mem, out in zip(outputs, self.outputs):
            mem[0] = out.get_value(return_internal_type=True, borrow=True)



if theano.sandbox.cuda.cuda_available:
    from theano.gof import local_optimizer
    from theano.sandbox.cuda.opt import register_opt
    from theano.sandbox.cuda.basic_ops import gpu_from_host, host_from_gpu
    from theano.sandbox.cuda.type import CudaNdarrayType

    @register_opt()
    @local_optimizer([])
    def local_gpu_forloop(node):
        if isinstance(node.op, forloop):
            sw = False
            for inp in node.inputs:
                if inp.owner and inp.owner.op == host_from_gpu:
                    sw = True
            if sw:
                inps = node.inputs
                nw_inps = []
                for inp in inps:
                    if not isinstance(inp.type, CudaNdarrayType):
                        nw_inps.append(gpu_from_host(inp))
                    else:
                        nw_inps.append(inp)
                new_outs = node.op(*nw_inps)
                return [host_from_gpu(x) for x in new_outs]
            else:
                return False


def print_time(secs):
    if secs < 120.:
        return '%6.3f sec' % secs
    elif secs <= 60 * 60:
        return '%6.3f min' % (secs / 60.)
    else:
        return '%6.3f h  ' % (secs / 3600.)


def print_mem(context=None):
    if theano.sandbox.cuda.cuda_enabled:
        rvals = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        # Avaliable memory in Mb
        available = float(rvals[0]) / 1024. / 1024.
        # Total memory in Mb
        total = float(rvals[1]) / 1024. / 1024.
        if context == None:
            print ('Used %.3f Mb Free  %.3f Mb, total %.3f Mb' %
                   (total - available, available, total))
        else:
            info = str(context)
            print (('GPU status : Used %.3f Mb Free %.3f Mb,'
                    'total %.3f Mb [context %s]') %
                    (total - available, available, total, info))


def const(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))

def gaussian(size, sigma):
    # Function borrowed from bengioe_util

    height = float(size)
    width = float(size)
    center_x = width  / 2. + 0.5
    center_y = height / 2. + 0.5

    gauss = numpy.zeros((height, width))

    for i in xrange(1, int(height) + 1):
        for j in xrange(1, int(height) + 1):
            x_diff_sq = ((float(j) - center_x)/(sigma*width)) ** 2.
            y_diff_sq = ((float(i) - center_y)/(sigma*height)) ** 2.
            gauss[i-1][j-1] = numpy.exp( - (x_diff_sq + y_diff_sq) / 2.)

    return gauss

def lcn_std_diff(x,size=9):
    # Function borrowed from bengioe_util
    p = x.reshape((1,1,48,48))
    #p = (p-TT.mean(p))/T.std(p)
    g = gaussian(size,1.591/size)
    g/=g.sum()
    g = numpy.float32(g.reshape((1,1,size,size)))
    mean = TT.nnet.conv.conv2d(p,TT.constant(g),
                              (1,1,48,48),
                              (1,1,size,size),
                              'full').reshape((48+size-1,)*2)
    mean = mean[size/2:48+size/2,
                size/2:48+size/2]
    meansq = TT.nnet.conv.conv2d(TT.sqr(p),TT.constant(g),
                                (1,1,48,48),
                                (1,1,size,size),
                                'full').reshape((48+size-1,)*2)
    meansq = meansq[size/2:48+size/2,
                    size/2:48+size/2]
    var = meansq - TT.sqr(mean)
    var = TT.clip(var, 0, 1e30)
    std = TT.sqrt(var)
    std = TT.clip(std, TT.mean(std), 1e30)
    out = (p - mean) / std
    return out - out.min()

def lcn(x,ishape,size=9):
    # Function borrowed from bengioe_util
    """
    expects x to be tensor{3|4}, the first dimension being the number
    of images, and the two last the shape of the image (which should be
    given anyways for optimization purposes
    """
    inshape = (x.shape[0],1,ishape[0],ishape[1])
    p = x.reshape(inshape)
    #p = (p-TT.mean(p))/T.std(p)
    g = gaussian(size,1.591/size)
    g/=g.sum()
    g = numpy.float32(g.reshape((1,1,size,size)))
    mean = TT.nnet.conv.conv2d(p,TT.constant(g),
                              None,
                              (1,1,size,size),
                              'full').reshape(
                                  (x.shape[0],1)+(ishape[0]+size-1,)*2)
    mean = mean[:,:,
                size/2:ishape[0]+size/2,
                size/2:ishape[1]+size/2]
    v = (p - mean)#.dimshuffle('x','x',0,1)
    var = TT.nnet.conv.conv2d(TT.sqr(v),TT.constant(g),
                             None,
                             (1,1,size,size),
                             'full').reshape(
                                  (x.shape[0],1)+(ishape[0]+size-1,)*2)
    var = var[:,:,
              size/2:ishape[0]+size/2,
              size/2:ishape[1]+size/2]
    std = TT.sqrt(var)
    std_mean = TT.mean(TT.mean(std,axis=3),axis=2).dimshuffle(0,1,'x','x')
    out = v / TT.maximum(std,std_mean)
    return (out + 2.5 )/5# - out.min()

def softmax(x):
    e = TT.exp(x)
    return e / TT.sum(e, axis=1).dimshuffle(0, 'x')
