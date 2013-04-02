from pylearn2.config import yaml_parse
import sys

dataset = yaml_parse.load_path(sys.argv[1])

var = dataset.X.var(axis=0)

print (var.min(), var.mean(), var.max())

mn = dataset.X.mean(axis=0)

print (mn.min(), mn.mean(), mn.max())
