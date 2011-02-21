# Clean-up of main.py code to allow embedding the alc calculation in a training algorithm
import numpy as np
from make_learning_curve import make_learning_curve
from alc import alc
import time
""" Compute the score as done on the website on a dataset with the corresponding labels
#
# Inputs:
#   dataset nxd    numpy matrix with n the nb of examples and d the number of features
#   labels  nxc    'one-hot' encoded numpy matrix with n the nb of examples and c
#                  the number of classes.
#   min_repeat     minimum number of repetitions for a specific number of examples
#   max_repeat     maximum number of repetitions for a specific number of examples
#   ebar           error bar for early termination of repetitions in case the error is sufficiently low
#   max_point_num  maximum number of points on the learning curve
#   debug          activate printing of error messages
#   
# Output:
#                  Area under the learning curve result
#
"""
   
def score(dataset, labels, 
            min_repeat=10, 
            max_repeat=500, 
            ebar=0.01, 
            max_point_num=7, 
            debug=False):

    #Make the learning curve
    x, y, e  = make_learning_curve(
                dataset, 
                labels, 
                min_repeat, 
                max_repeat, 
                ebar, 
                max_point_num,
                debug,
                useRPMat=True # Whether we should use a precalculated permutation matrix
               )
    # Compute the (normalized) area under the learning curve
    # returns the ALC and the last AUC value
    print 'auc evolution',y
    return alc(x, y)

if __name__ == "__main__":
    import pylearn.datasets.utlc as pdu
    # Test on validation data
    (dataset_devel, dataset_valid, dataset_test) = \
        pdu.load_ndarray_dataset("ule", normalize=False, transfer=False) 
    (labels_devel, labels_valid, labels_test)  = pdu.load_ndarray_label("ule")
    tic = time.clock()

    dataset = np.vstack((dataset_valid, dataset_test))
    #_labels_valid = np.hstack((labels_valid, np.zeros((labels_valid.shape[0],labels_test.shape[1]))))
    _labels_valid = np.hstack((np.ones((labels_valid.shape[0],1)), np.zeros((labels_valid.shape[0],1))))
    #_labels_test = np.hstack((np.zeros((labels_test.shape[0],labels_valid.shape[1])),labels_test))
    _labels_test = np.hstack((np.zeros((labels_test.shape[0],1)),np.ones((labels_test.shape[0],1))))
    labels = np.vstack((_labels_valid, _labels_test))

    print labels_valid
    print '... computing score on dataset of shape', dataset.shape,\
        'and labels of shape', labels.shape
    print score(dataset, labels)
    toc = time.clock()
    print 'computed in %f (min)'%((toc - tic)/60.)
