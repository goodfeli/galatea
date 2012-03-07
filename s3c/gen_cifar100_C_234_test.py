
for i in xrange(10):
    start = i  * 1000
    end = (i+1)  * 1000

    char = chr(ord('A')+i)

    fname = 'config/cifar100/extract/C_test_'+char+'_234.yaml'


    f = open(fname,'w')
    f.write("""!obj:galatea.s3c.extract_features.FeatureExtractor {
        "batch_size" : 1,
        "model_path" : "/RQexec/goodfell/C.pkl",
        "save_paths" : [ "/RQexec/goodfell/C_test_2_"""+char+""".npy",
        "/RQexec/goodfell/C_test_3_"""+char+""".npy",
        "/RQexec/goodfell/C_test_4_"""+char+""".npy" ] ,
        "pooling_region_counts" : [ 2,3,4],
        "feature_type" : "exp_h",
        "dataset_family" : galatea.s3c.extract_features.cifar100,
        "restrict" : [ %d,  %d ],
        "which_set" : "test"
}""" % (start, end))
    f.close()

assert end == 10000
