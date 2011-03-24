import zipfile
from tempfile import TemporaryFile
import numpy as N
import serialutil


k = 8

"""X = serialutil.load('tsne_terry.pkl')


valid_start = 4000
test_start = valid_start + 4096
test_end = test_start + 4096
assert test_end == X.shape[0]

valid = X[valid_start:test_start,:]
"""

valid = N.loadtxt('harry_best_valid.prepro')
valid_only = True


print 'warning, test is just dummy features'

valid_feat = N.zeros((valid.shape[0],valid.shape[0]))
test = valid_feat

for i in xrange(valid.shape[0]):
    print i
    diffs = valid - valid[i,:]
    dists = N.sqrt(N.square(diffs).sum(axis=1))

    vals = list(dists)
    vals = sorted(vals)
    thresh = vals[k]

    valid_feat[:,i] = dists < thresh


print valid_feat.sum(axis=0)


valid = valid_feat

print "Saving"
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

submission = zipfile.ZipFile("terry_tsne_"+str(k)+"nn.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()

