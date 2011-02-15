import cProfile
import pstats

from embed import score
import pylearn.datasets.utlc as pdu

def main():
    # Test on validation data
    (dataset_devel, dataset_valid, dataset_test) = \
        pdu.load_ndarray_dataset("ule", normalize=False, transfer=False) 
    (labels_devel, labels_valid, labels_test)  = pdu.load_ndarray_label("ule")

    print "Print computed score: "
    print score(dataset_valid, labels_valid)#, min_repeat=1, max_repeat=10)

cProfile.run('main()', 'scoreprof')

p = pstats.Stats('scoreprof')
p.sort_stats('time').print_stats(10)

