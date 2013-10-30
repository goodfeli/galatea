base = 'config/6x6/extract/G2_1800K'
pkl_path = '${GALATEA_PATH}/s3c/config/6x6/G2_1800K.pkl'
script1 = open(base+'_script_1.sh','w')
script2 = open(base+'_script_2.sh','w')
script3 = open(base+'_script_3.sh','w')

for i in xrange(10):
    start = i * 5000
    end = (i+1) * 5000

    char = chr(ord('A')+i)

    fname = base+'_'+char+'.yaml'

    if i < 3:
        script = script1
    elif i < 6:
        script = script2
    else:
        script = script3

    script.write('THEANO_FLAGS="device=gpu0" python extract_features.py '+fname+'\n')

    f = open(fname,'w')
    f.write("""!obj:galatea.s3c.extract_features.FeatureExtractor {
        "batch_size" : 1,
        "model_path" : "%s",         "pooling_region_counts": [3],
        "save_paths" : [ "${FEATURE_EXTRACTOR_YAML_PATH}.npy" ],
        "feature_type" : "exp_h",
        "dataset_family" : galatea.s3c.extract_features.cifar10,
        "restrict" : [ %d,  %d ],
        "which_set" : "train"
}""" % (pkl_path, start, end))
    f.close()

script1.close()
script2.close()
script3.close()
