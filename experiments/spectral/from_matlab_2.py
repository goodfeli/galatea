import zipfile
from tempfile import TemporaryFile
import numpy as N
from scipy import io

Y = io.loadmat('/u/goodfeli/rep.mat')
valid = Y['rep']
test = valid


valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)


submission = zipfile.ZipFile("matlab_2.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("harry_sda_valid.prepro", valid_file.read())
submission.writestr("harry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()



