import zipfile
from tempfile import TemporaryFile
import numpy as N
import sys

outpath = sys.argv[1]
arg1 = sys.argv[2]
arg2 = sys.argv[3]







valid_file = TemporaryFile()
test_file = TemporaryFile()

N.savetxt(valid_file, valid, fmt="%.3f")
N.savetxt(test_file, test, fmt="%.3f")

valid_file.seek(0)
test_file.seek(0)

name = name.replace('means','')
submission = zipfile.ZipFile(name+feature_type+".zip" , "w", compression=zipfile.ZIP_DEFLATED)

submission.writestr("terry_sda_valid.prepro", valid_file.read())
submission.writestr("terry_sda_final.prepro", test_file.read())

submission.close()
valid_file.close()
test_file.close()

