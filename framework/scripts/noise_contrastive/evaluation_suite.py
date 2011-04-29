import sys
from framework.utils import serial
from framework.config import yaml_parse

model = serial.load(sys.argv[1])
model.redo_theano()


for dataset_yaml_src in [model.dataset_yaml_src,model.dataset_yaml_src.replace('train','test')]:

    dataset = yaml_parse.load(dataset_yaml_src)

    dataset.reset_RNG()
    model.print_suite(dataset, 5, 1000,
            [ ("objective",model.error_func),
              ("data energy",model.E_d_func),
              ("corruption energy",model.E_c_func),
              ("misclassification rate",model.misclass_func) ] )


