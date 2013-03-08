from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class BagOfWords(DenseDesignMatrix):

    def __init__(self, X, words, files):
        self.words = words
        self.files = files

        super(BagOfWords, self).__init__(X)


