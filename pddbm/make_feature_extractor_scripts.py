import sys
from pylearn2.utils import serial

ignore, model_path, script_dir = sys.argv

serial.mkdir(script_dir)

chunk_size = 1000

m = 50000

assert m % chunk_size == 0

num_chunks = m / chunk_size

for i in xrange(num_chunks):
    start = i * chunk_size
    stop = (i+1)*chunk_size
    name = 'chunk_%d.yaml' % i
    f = open(script_dir + '/' + name, 'w')
    f.write("""!obj:galatea.pddbm.extract_features.FeatureExtractor {
            batch_size : 1,
            model_path : %(model_path)s,
            pooling_region_counts : [ 3 ],
            save_paths : [ %(script_dir)s/chunk_%(i)d.npy ],
            feature_type : "exp_h,exp_g",
            dataset_family : galatea.pddbm.extract_features.cifar100,
            which_set : "train",
            restrict : [ %(start)d, %(stop)d ]
        }""" % locals() )
    f.close()


