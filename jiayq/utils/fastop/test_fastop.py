from _fastop import *

threshold = 1e-14

class TestFastmax:
    def setup(self):
        pass
        
    def teardown(self):
        pass

    def test_fastmeanstd(self):
        matrix = np.random.rand(2,3)
        meanstd = fastmeanstd(matrix[0])
        print matrix[0]
        print meanstd
        print np.mean(matrix[0]),np.std(matrix[0])
        assert np.abs(meanstd[0] - np.mean(matrix[0])) <= threshold
        assert np.abs(meanstd[1] - np.std(matrix[0])) <= threshold
        meanstd = fastmeanstd(matrix[1])
        assert np.abs(meanstd[0] - np.mean(matrix[1])) <= threshold
        assert np.abs(meanstd[1] - np.std(matrix[1])) <= threshold
        meanstd = fastmeanstd(matrix[0,:-1])
        assert np.abs(meanstd[0] - np.mean(matrix[0,:-1])) <= threshold
        assert np.abs(meanstd[1] - np.std(matrix[0,:-1])) <= threshold
        meanstd = fastmeanstd(matrix[1,:-1])
        assert np.abs(meanstd[0] - np.mean(matrix[1,:-1])) <= threshold
        assert np.abs(meanstd[1] - np.std(matrix[1,:-1])) <= threshold

    def test_fastmax(self):
        matrix = np.random.rand(100,101)
        rowids = np.random.randint(2,size=100)
        while np.sum(rowids) == 0:
            rowids = np.random.randint(1,size=100)
        
        out_correct = np.max(matrix[rowids==1], axis=0)
        temp = np.zeros((2,101))
        # test two types of input
        out_align = temp[0]
        out_misalign = temp[1]
        out_auto = fastmaxm(matrix,rowids)
        fastmaxm(matrix,rowids,out_align)
        fastmaxm(matrix,rowids,out_misalign)
        assert np.all(np.abs(out_correct - out_auto) <= threshold)
        assert np.all(np.abs(out_correct - out_align) <= threshold)
        assert np.all(np.abs(out_correct - out_misalign) <= threshold)
        rowids = np.flatnonzero(rowids)
        out_auto = fastmaxm(matrix,rowids)
        fastmaxm(matrix,rowids,out_align)
        fastmaxm(matrix,rowids,out_misalign)
        assert np.all(np.abs(out_correct - out_auto) <= threshold)
        assert np.all(np.abs(out_correct - out_align) <= threshold)
        assert np.all(np.abs(out_correct - out_misalign) <= threshold)

    def test_fastsum(self):
        matrix = np.random.rand(100,101)
        rowids = np.random.randint(2,size=100)
        while np.sum(rowids) == 0:
            rowids = np.random.randint(1,size=100)
        
        out_correct = np.sum(matrix[rowids==1], axis=0)
        temp = np.zeros((2,101))
        # test two types of input
        out_align = temp[0]
        out_misalign = temp[1]
        out_auto = fastsumm(matrix,rowids)
        fastsumm(matrix,rowids,out_align)
        fastsumm(matrix,rowids,out_misalign)
        assert np.all(np.abs(out_correct - out_auto) <= threshold)
        assert np.all(np.abs(out_correct - out_align) <= threshold)
        assert np.all(np.abs(out_correct - out_misalign) <= threshold)
        rowids = np.flatnonzero(rowids)
        out_auto = fastsumm(matrix,rowids)
        fastsumm(matrix,rowids,out_align)
        fastsumm(matrix,rowids,out_misalign)
        assert np.all(np.abs(out_correct - out_auto) <= threshold)
        assert np.all(np.abs(out_correct - out_align) <= threshold)
        assert np.all(np.abs(out_correct - out_misalign) <= threshold)

    def test_normalizev(self):
        v = np.random.rand(10)
        m = np.mean(v)
        std = np.std(v)
        v_correct = (v - m) / std
        normalizev(v,m,std)
        print np.max(v-v_correct), 'float64 eps is:', threshold
        assert np.all(np.abs(v - v_correct) <= threshold)
        
    def test_fastcenters(self):
        k = 10
        matrix = np.random.rand(1000,10)
        z = np.random.randint(k,size=1000)
        centers = np.empty((k, 10))
        for q in range(k):
            center_mask = np.flatnonzero(z == q)
            if len(center_mask) > 0:
                centers[q] = np.mean(matrix[center_mask], axis=0)
        centers_fastop = fastcenters(matrix,z,k)[0]
        for q in range(k):
            center_mask = np.flatnonzero(z==q)
            if len(center_mask) > 0:
                print 'debug:', np.vstack((centers[q],centers_fastop[q]))
                assert np.all(np.abs(centers[q] - centers_fastop[q]) <= threshold)
        
    def test_fastmaximums(self):
        k = 10
        matrix = np.random.rand(1000,10)
        z = np.random.randint(k,size=1000)
        centers = np.empty((k, 10))
        for q in range(k):
            center_mask = np.flatnonzero(z == q)
            if len(center_mask) > 0:
                centers[q] = np.max(matrix[center_mask], axis=0)
        centers_fastop = fastmaximums(matrix,z,k)[0]
        for q in range(k):
            center_mask = np.flatnonzero(z==q)
            if len(center_mask) > 0:
                print 'debug:', np.vstack((centers[q],centers_fastop[q]))
                assert np.all(np.abs(centers[q] - centers_fastop[q]) <= threshold)
