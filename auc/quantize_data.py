import pylearn.datasets.utlc as pdu
import numpy as N

def quantize_data(dataset):
    """ not actually implemented yet, just checks that data is aleady quantized.
        if it isn't quantized, just kills the program """

    # 'TODO-- should quantize_data change dtype to uint16 or something like that?'

    if dataset.quantized:
        print 'dataset already quantized'
        return dataset

    dataset.quantized = True

    for set_name in dataset.X0:
        #don't process devel, it is too big
        #(according to original matlab sample code)
        if set_name == 'devel': 
            continue
        #endif

        X = dataset.X0[set_name]

        mini = X.min()
        maxi = X.max()

        #print X.dtype
        if mini >= 0 and maxi <= 999 and (str(X.dtype).find('int') != -1 or N.all(X == X.round())):
            print 'dataset already quantized'
            return dataset
        #endif

        assert False #unimplemented
    #end for loop on set_name

    return dataset
#end def of quantize_data




