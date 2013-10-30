from pylearn2.utils import serial
import sys

model_path, kmeans_path = sys.argv[1:]

print 'loading model'
model = serial.load(model_path)
print 'loading kmeans'
kmeans = serial.load(kmeans_path)

from galatea.s3c.s3c_dataset import S3C_Dataset

from pylearn2.config import yaml_parse

print 'loading dataset'
raw = yaml_parse.load(model.dataset_yaml_src)

print 'making transformer dataset'
dataset = S3C_Dataset(raw = raw, transformer = model)

from pylearn2.gui.get_weights_report import get_weights_report

print 'making weights report'
pv = get_weights_report(model = kmeans, dataset = dataset)

pv.show()
