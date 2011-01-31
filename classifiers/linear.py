# Linear Classifier template object
import numpy

class LinearClassifier(object):
    """Template for linear classifier objects"""

    def __init__(self):
        """
        """
        self.b0 = 0
        self.W  = numpy.array([]);

    def one_class_classifier(self, X, Y):
        """
        If training example(s) are from a single class, adequately
        set the parameters of the classifier, otherwise do nothing.

        :type   X: numpy.array
        :param  X: training examples
        :type   Y: numpy.array
        :param  Y: training labels
        """
        lbl = numpy.unique(Y)
        if len(lbl) == 1:
            # We define an hyperplane centered at the origin
            # with a normal parallel to the vector pointing
            # to the mean, in the direction of positive labels
            self.W = mean(X, axis=0) * lbl 


    def train(self, X, Y):
        """
        Train the linear classifier on the provided labeled examples.
        The linear classifier should support being trained with a single
        example or examples from a single class.

        Labels should either be +1 for the positive class or -1 for the
        negative class.

        :type  X: numpy.array
        :param X: training examples
        :type  Y: numpy.array
        :param Y: training labels 
        """
        self.one_class_classifier(X, Y)

        # Check if there was only one class of examples
        if len(self.W) != 0:
            return

        if pd_check(X):
            # If the data matrix is a kernel matrix, 
            # follow the model f(X) = X*W' + b0, where 
            # X is in fact a kernel matrix of the type X*X'.
            #   (self.W, self.b0)=some_first_function(X, Y);
            # Expect that the test data will be a matrix Xte*X'.
            pass
        else:
            # Compute the weight and bias of the model f(X) = X*W' + b0
            #   (self.W, self.b0)=some_first_function(X, Y);
            pass

    def test(self, X):
        """
        Returns the predicted labels for some test examples.

        :type  X:  numpy.array 
        :param X:  test data
        """
        eps = numpy.MachAr().eps
        
        if X.shape[1] != self.W.shape[0]:
            preds = numpy.array([])
        else:
            preds = numpy.dot(X, self.W.T) + self.b0

        # Remove ties (the negative class is usually most abundant):
        zeroes_idx = (preds == 0)
        preds[zeroes_idx] = -eps

        return preds


