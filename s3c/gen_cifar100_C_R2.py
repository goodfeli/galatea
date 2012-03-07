
for i in xrange(5):
    start = i * 2 * 5000
    end = (i+1) * 2 *5000

    char = chr(ord('A')+i)

    fname = 'config/cifar100/extract/C_briaree_'+char+'_R2.yaml'


    f = open(fname,'w')
    f.write("""!obj:galatea.s3c.ef2.FeatureExtractor {
        "batch_size" : 1,
        "model_path" : "/RQexec/goodfell/C.pkl",
        "save_path" :  "/RQexec/goodfell/C_briaree_4_"""+char+""".npy"  ,
        "feature_type" : "exp_h",
        "dataset_family" : galatea.s3c.extract_features.cifar100,
        "restrict" : [ %d,  %d ],
        "which_set" : "train"
}""" % (start, end))
    f.close()

