import zipfile
from tempfile import TemporaryFile
import numpy as N


valid = N.load('/u/goodfeli/avicenna_bg2/valid_rbm.npy')
test = N.load('/u/goodfeli/avicenna_bg2/test_rbm.npy')

valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)


submission = zipfile.ZipFile("/u/goodfeli/avicenna_bg2/rbm.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()



