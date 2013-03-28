from _fastmax import *

class TestFastmax:
    def setup(self):
        pass
        
    def teardown(self):
        pass

    def test_fastmax(self):
        matrix = np.random.rand(100,101)
        rowids = np.random.randint(100,size=10)
        out_correct = np.max(matrix[rowids], axis=0)
        temp = np.zeros((2,101))
        out_align = temp[0]
        out_misalign = temp[1]
        fastmaxm(matrix,rowids,out_align)
        fastmaxm(matrix,rowids,out_misalign)
        assert np.all((out_correct - out_align) < np.finfo(np.float64).eps)
        assert np.all((out_correct - out_misalign) < np.finfo(np.float64).eps)

    def test_normalizev(self):
        v = np.random.rand(10)
        m = np.mean(v)
        std = np.std(v)
        v_correct = (v - m) / std
        normalizev(v,m,std)
        print v
        print v_correct
        assert np.all((v - v_correct) < np.finfo(np.float64).eps)
