
for i in xrange(10):
    start = i * 5000
    end = (i+1) * 5000

    char = chr(ord('A')+i)

    fname = 'config/cifar100/extract/B1_briaree_'+char+'.yaml'


    f = open(fname,'w')
    f.write("""!obj:galatea.s3c.extract_features_rand_pool_python_loop.FeatureExtractor {
        "batch_size" : 1,
        "model_path" : "/RQexec/goodfell/B1.pkl",
        "save_path" :  "/RQexec/goodfell/B1_briaree_"""+char+""".npy" ,
        "feature_type" : "exp_h",
        "dataset_family" : galatea.s3c.extract_features.cifar100,
        "restrict" : [ %d,  %d ],
        "which_set" : "train"
}""" % (start, end))
    f.close()

