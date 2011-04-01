import zipfile
from tempfile import TemporaryFile
import numpy as N
from scipy import io

Y = io.loadmat('Y.mat')
Y = Y['Y']

assert Y.shape[0] == 4000 + 4096 + 4096

valid = Y[4000:8096,:]
test  = Y[8096:,:]

valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)


submission = zipfile.ZipFile("matlab.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()



