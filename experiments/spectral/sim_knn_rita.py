import zipfile
from tempfile import TemporaryFile
import numpy as N
import serialutil
from auc import embed

k = 1000.


valid = N.loadtxt('/u/goodfeli/rita/rita_sda_valid.prepro')
test  = N.loadtxt('/u/goodfeli/rita/rita_sda_final.prepro')


#valid = serialutil.load('/data/lisatmp/ift6266h11/rita-pca-lcn/newdata_pca_lcn/valid1000.pkl')
#test =  serialutil.load('/data/lisatmp/ift6266h11/rita-pca-lcn/newdata_pca_lcn/test1000.pkl' )

#valid = N.concatenate((valid,test),axis=0)


print 'warning, test is just dummy features'

valid = N.concatenate((valid,test),axis=0)

valid_feat = N.zeros((valid.shape[0],valid.shape[0]))

for i in xrange(valid.shape[0]):
    print i
    diffs = valid - valid[i,:]
    dists = N.sqrt(N.square(diffs).sum(axis=1))

    #vals = list(dists)
    #vals = sorted(vals)
    #thresh = vals[k]

    #valid_feat[:,i] = dists < thresh
    valid_feat[:,i] = N.exp(-dists ** 2. / k )

print valid_feat.sum(axis=0)


valid = valid_feat[0:4096,:]
#test = valid_feat[4096:,:]
test = valid

print k
fuckyou =  N.concatenate((valid_feat, N.zeros((4096*2,1))),axis=1)
print fuckyou.shape
print embed.score( fuckyou
        , N.concatenate((N.zeros((4096,1)),N.ones((4096,1)))) )




print "Saving"
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

submission = zipfile.ZipFile("rita_cae3_parzen_"+str(k)+"nn.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("rita_sda_valid.prepro", valid_file.read())
submission.writestr("rita_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()

