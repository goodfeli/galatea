
for alpha, alpha_str in [ (0.1, '0_1'), (0.25,'0_25'), (0.5, '0_5'), (1, '1') ]:

    f = open('omp_2_train_'+alpha_str+'.yaml','w')

    f.write("""!obj:galatea.s3c.extract_features_omp.FeatureExtractor {
    "chunk_size" : 10000,
    "batch_size" : 1,
    "save_paths" :  [ "/RQexec/goodfell/omp_2_train_"""+alpha_str+""".npy"  ] ,
    "alpha" : """+str(alpha)+""",
    "num_bases" : 1600,
    "pooling_region_counts" : [2],
    "dataset_family" : galatea.s3c.extract_features.cifar100,
    "which_set" : "train"
}
""")

    f.close()



    f = open('omp_3_800_train_'+alpha_str+'.yaml','w')

    f.write("""!obj:galatea.s3c.extract_features_omp.FeatureExtractor {
    "chunk_size" : 10000,
    "batch_size" : 1,
    "save_paths" :  [ "/RQexec/goodfell/omp_2_train_"""+alpha_str+""".npy"  ] ,
    "alpha" : """+str(alpha)+""",
    "num_bases" : 800,
    "pooling_region_counts" : [3],
    "dataset_family" : galatea.s3c.extract_features.cifar100,
    "which_set" : "train"
}
""")

    f.close()
