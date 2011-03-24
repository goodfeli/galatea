import zipfile
from tempfile import TemporaryFile
import numpy as N


k = 100000.


valid = N.loadtxt('harry_best_valid.prepro')
valid_only = True

"""
X = serialutil.load('tsne_terry.pkl')


valid_start = 4000
test_start = valid_start + 4096
test_end = test_start + 4096
assert test_end == X.shape[0]

valid = X[valid_start:test_start,:]
test = X[test_start:,:]



print 'warning, test is just dummy features'
"""


valid -= valid.mean(axis=0)
valid /= valid.std(axis=0)


valid_feat = N.zeros((valid.shape[0],valid.shape[0]))


for i in xrange(valid.shape[0]):
    diffs = valid - valid[i,:]
    sqdists = N.square(diffs).sum(axis=1)

    valid_feat[:,i] = N.exp(-sqdists/k)
    print (i,valid_feat[:,i].sum())


valid = valid_feat
test = valid_feat

print "Saving"
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

submission = zipfile.ZipFile("parzen_harry_"+str(k)+".zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()

