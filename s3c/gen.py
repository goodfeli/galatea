#script = open('config/6x6/extract/H_interm.sh','w')

for i in xrange(10):
    start = i * 5000
    end = (i+1) * 5000

    char = chr(ord('A')+i)

    fname = 'config/6x6/extract/H_interm_exp_h_3_train_cpu_'+char+'.yaml'

    #script.write('THEANO_FLAGS="device=gpu0" python extract_features.py '+fname+'\n')

    f = open(fname,'w')
    f.write("""!obj:galatea.s3c.extract_features.FeatureExtractor {
        "batch_size" : 1,
        "model_path" : "${GALATEA_PATH}/s3c/config/6x6/H_interm_cpu.pkl",         "pooling_region_counts": [3],
        "save_paths" : [ "${FEATURE_EXTRACTOR_YAML_PATH}.npy" ],
        "feature_type" : "exp_h",
        "dataset_name" : "cifar10",
        "restrict" : [ %d, %d ],
        "which_set" : "train"
}""" % (start, end))
    f.close()

#script.close()
