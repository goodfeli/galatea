"""
Load a sparse UTLC dataset and saves it in the filetensor format.
"""

import os
import sys

import numpy
import pylearn.io.filetensor
import scipy.sparse

import functools as fc

# temporaire
ROOT_PATH = '/part/02/Tmp/saintjaf/utlc/raw'

datasets_avail = ["avicenna","harry","rita","sylvester","ule"]
subset = ["devel", "valid", "final"]

datasets_shape = {
        "avicenna_devel" : 150205,
        "harry_devel" : 69652,
        "rita_devel" : 111808,
        "sylvester_devel" : 572820,
        "terry_devel" : 217034,
        "ule_devel" : 26808 }

# normalize functions
def normalize_gaussian(mean, std, set):
    return (set - mean) / std

def normalize_maximum(max, set):
    return set / max

# statistics
avi_mean = 514.62248464301717
avi_std  = 6.901698313710936
harry_std = 1.0
rita_max = 290.0
syl_mean = 235.22886219217503
syl_std  = 173.08160705637636
ule_max = 255.0

datasets_normalizer= {
        "avicenna"  : fc.partial(normalize_gaussian, avi_mean, avi_std),
        "harry"     : fc.partial(normalize_maximum, harry_std),
        "rita"      : fc.partial(normalize_maximum, rita_max),
        "sylvester" : fc.partial(normalize_gaussian, syl_mean, syl_std),
        "ule"       : fc.partial(normalize_maximum, ule_max) }

def load_dataset(name, subset, permute_train=True, normalize=True):
    if not os.path.exists(os.path.join(ROOT_PATH,name+'_text')):
        raise Exception("The directory with the original data does not exists")

    data = numpy.fromfile(os.path.join(ROOT_PATH,name+'_text',name+'_'+subset+'.data'),
                           dtype=numpy.float32, sep=' ')

    size = 4096
    if subset == "devel":
        size = datasets_shape[name + "_devel"]
    data = data.reshape(size, data.shape[0]/size)

    if permute_train:
        rng = numpy.random.RandomState([1,2,3])
        perm = rng.permutation(data.shape[0])
        data = data[perm]

    if normalize:
        data = datasets_normalizer[name](data)

    return data

def write_dataset(name, subset, dataset):
    print "Wrinting dataset", name
    pylearn.io.filetensor.write(open(name+'_'+subset+'.ft','wb'),dataset)

todo = sys.argv[1:]
if len(sys.argv)<=1 or any([d not in datasets_avail for d in todo]):
    print "Usage: to_npy.py [avicenna, harry, rita, sylvester, ule]"
    quit(1)

print "Will process datasets:", todo
for data in todo:
    for s in subset:
        write_dataset(data, s, load_dataset(data,s))
