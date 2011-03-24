import zipfile
from tempfile import TemporaryFile
import numpy as N
import serialutil

X = serialutil.load('tsne_terry.pkl')

valid_start = 4000
test_start = valid_start + 4096
test_end = test_start + 4096
assert test_end == X.shape[0]

valid = X[valid_start:test_start,:]
test = X[test_start:,:]

print "Saving"
valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

submission = zipfile.ZipFile("terry_tsne.zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()

