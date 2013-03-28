from pylearn2.utils import serial
from pylearn2.datasets import preprocessing



patch_shape = (6,6)
ipt = '/data/lisatmp/goodfeli/darpa_imagenet_patch_%dx%d_train.pkl' % (patch_shape[0], patch_shape[1])
opt = '/data/lisatmp/goodfeli/darpa_imagenet_patch_%dx%d_train_preprocessed.pkl' % (patch_shape[0], patch_shape[1])
opt_npy = '/data/lisatmp/goodfeli/darpa_imagenet_patch_%dx%d_train_preprocessed.npy' % (patch_shape[0], patch_shape[1])
opt_prepro = '/data/lisatmp/goodfeli/darpa_imagenet_patch_%dx%d_train_preprocessor.npy' % (patch_shape[0], patch_shape[1])

train = serial.load(ipt)

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA())


train.apply_preprocessor(preprocessor = pipeline, can_fit = True)

train.use_design_loc(opt_npy)
serial.save(opt,train)
serial.save(opt_prepro,pipeline)
