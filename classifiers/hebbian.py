from linear import LinearClassifier
from pd_check import pd_check
import numpy

class Hebbian(LinearClassifier):
    """
    Hebbian classifier, used for official evaluation. The learned hyper-plan's 
    is normal to the line joining the mean positions of the positive and 
    negative training examples.
    """

    def __init__(self):
        LinearClassifier.__init__(self)

    def train(self, X, Y):
        """
        Find the hyper-plan normal to the vector joining the mean positions of
        the positive and negative training examples.

        If non-kernelized data is used for training, the plan will intersect the
        mean position of all the training examples.  However, if kernelized data
        is used, the plan will intersect the origin.

        kernelized data means that X = dot(Xtr,Xtr.T)
        non-kernelized data means that X = Xtr
        where Xtr are the training examples

        :type  X: numpy.array
        :param X: training data
        :type  Y: numpy.array
        :param Y: training labels
        """

        (n,d) = X.shape

        posidx = Y > 0 
        negidx = Y < 0

        eps = numpy.MachAr().eps

        if pd_check(X):
            self.W = numpy.zeros(Y.shape[0])
            self.W[posidx] = 1.0/(sum(posidx) + eps)
            self.W[negidx] = -1.0/(sum(negidx) + eps)
        else:
            Mu1 = numpy.zeros(d)
            Mu2 = numpy.zeros(d)

            if sum(posidx) > 0:
                Mu1 = numpy.mean(X[posidx,:], axis=0)

            if sum(negidx) > 0:
                Mu2 = numpy.mean(X[negidx,:], axis=0)

            self.W = Mu1 - Mu2
            self.b0 = -numpy.dot(self.W,(Mu1+Mu2))/2

if __name__ == "__main__":
    def test_case(name, data, test_data):
        print "Test case '" + name + "'"

        X = data[:,:-1]
        Y = data[:,-1]

        print "** Non-kernelized version **"
        classifier = Hebbian()
        #print "before training:"

        #print classifier.b0
        #print classifier.W

        classifier.train(X,Y)
        print "after training:"
        print classifier.b0
        print classifier.W

        print "prediction:"
        print classifier.test(test_data)

        print "** Kernelized version **"
        classifier = Hebbian()

        #print "before training:"
        #print classifier.b0
        #print classifier.W

        if not pd_check(X):
            # Do not "kernelize" the data a second time
            # if it has been detected as such
            classifier.train(numpy.dot(X,X.T), Y)
        else:
            classifier.train(X, Y)

        print "after training:"
        print classifier.b0
        print classifier.W

        print "prediction:"
        print classifier.test(numpy.dot(test_data, X.T))

        print "End of '" + name + "'"
        print ""


    test_data = numpy.array([[0,1], 
                             [0,0]])
    
    # Bug in the matlab version:
    # makes the algorithm detect a positive definite matrix
    # in the non-kernelized version leading to divergence
    # between the results for the non-kernelized and the kernelized
    # version
    test_case("Bug", 
              numpy.array([[0.5, 1.5, -1],
                           [1.5, 0.5, 1]]), 
              test_data)
    
    # If the mean position of all the training examples
    # is not at the origin, the kernelized and non-kernelized
    # version won't give the same results. The following
    # example has been tested with the matlab script and gives
    # the same result:
    test_case("Mean not at the origin", 
              numpy.array([[0, 2, -1],
                           [1, 1,  1]]),
              test_data)

    # The following example shows that the results
    # for the kernelized and non-kernelized version are the same.
    # I double each example to force the non-kernelized version to be used. 
    test_case("Simple example", 
              numpy.array([[0.5,1.5,-1], 
                           [1.5,0.5,1],
                           [0.5,1.5,-1],
                           [1.5,0.5,1]]),
              test_data)


