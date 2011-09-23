from pylearn2.utils import serial
from pylearn2.datasets.cos_dataset import CosDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

d = CosDataset()

X = d.get_batch_design(10)

nd = DenseDesignMatrix(X)

serial.save('tiny_datset.pkl',nd)
