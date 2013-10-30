import sys
import SkyNet
from pylearn2.utils import serial
from scipy import io

job_name = sys.argv[1]

SkyNet.set_job_name(job_name)

components = SkyNet.get_dir_path('components')

pca_model = serial.load(components+'/pca_model.pkl')
whitener = serial.load(components+'/whitener.pkl')
W = serial.load(components+'/W.pkl')

d = {
        'pca_basis' : pca_model.get_weights(),
        'W'         : W,
        'whitening_basis' : whitener.get_weights(),
        'mu'        : pca_model.mean.get_value(),
        'mu2'       : whitener.mean.get_value()
    }


io.savemat('matlab_dump.mat', d)
