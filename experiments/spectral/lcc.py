import zipfile
from tempfile import TemporaryFile
import numpy as N
from scipy import io
from framework.datasets import matlab_dataset
from framework.utils import serial

Xvalid = matlab_dataset.MatlabDataset('/u/goodfeli/ift6266h11/experiments/fast_sc/data/terry_pca_512.mat','valid').get_design_matrix()
Xtest = matlab_dataset.MatlabDataset('/u/goodfeli/ift6266h11/experiments/fast_sc/data/terry_pca_512.mat','test').get_design_matrix()

Xvalid = Xvalid[0:4096,:]
Xtest = Xtest[0:4096,:]

assert Xvalid.shape[0] == 4096

model = serial.load('/u/goodfeli/ift6266h11/framework/scripts/lcc_terry_2.pkl')

model.coeff /= 100.

def process(X):
    nc = model.get_output_channels()
    m = X.shape[0]

    rval = N.zeros((m,nc))

    for i in xrange(m):
        rval[i,:] =  10000. * model.optimize_gamma(X[i,:])
    #

    return rval
#


valid = process(Xvalid)

print 'valid'
s = valid.std(axis=0)
print 'std: '+str((s.min(),s.mean(),s.max()))
if N.any(N.isnan(valid)):
    print 'there are NaNs'
if N.any(N.isinf(valid)):
    print 'there are infs'


test  = process(Xtest)

valid = valid[:,s != 0.0]
test = test[:,s != 0.0]

s = valid.std(axis=0)
print 'std: '+str((s.min(),s.mean(),s.max()))


valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)


submission = zipfile.ZipFile("terry_lcc.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_valid.prepro", valid_file.read())
submission.writestr("terry_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()



