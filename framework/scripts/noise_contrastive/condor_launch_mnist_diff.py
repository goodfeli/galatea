import numpy as N
import SkyNet

job_name = "local_noise_rbm_mnist_diff"

SkyNet.set_job_name(job_name)
configsDir = SkyNet.get_dir_path('configs')
modelsDir = SkyNet.get_dir_path('models')

rng = N.random.RandomState([1,2,3])

command = SkyNet.get_user_dir(tmp = False, force = SkyNet.cluster) + '/ift6266h11/framework/scripts/train.py '+configsDir+'/config_"{{'

first = True

num_jobs = 50

for i in xrange(num_jobs):

    if not first:
        command += ','
    ""

    first = False

    nhid = rng.randint(400,784)
    irange = rng.uniform(1e-3,.05)
    init_beta = N.exp(rng.uniform(N.log(1e-3),N.log(1.)))
    enc_weight_decay = N.exp(rng.uniform(N.log(1e-6),N.log(3e-2)))
    learning_rate = N.exp(rng.uniform(N.log(1e-4),N.log(1e-1) ) )
    fold_biases = rng.uniform(0,1) > 0.5
    batch_size = rng.randint(1,10)
    min_misclass = rng.uniform(0.01,0.2)
    max_misclass = rng.uniform(min_misclass+0.01,0.5)

    #if these are too big, we get stuck with beta being 1e-20
    beta_scale_up = N.exp(rng.uniform(N.log(1.00001),N.log(1.1)))
    beta_scale_down = N.exp(rng.uniform(N.log(.9),N.log(.9999999)))
    time_constant = N.exp(rng.uniform(N.log(1e-3),N.log(.05)))

    save_path = modelsDir + '/model_'+str(i)+'.pkl'

    config = """
!obj:framework.scripts.train.Train {
    "dataset": !pkl: "/data/lisatmp/goodfeli/mnist_preprocessed_train.pkl",
    "model": !obj:framework.models.local_noise_rbm.LocalNoiseRBM {
                "nvis" : 784,
                "nhid" : %(nhid)d,
                "init_bias_hid" : 0.0,
                "irange"  : %(irange)f,
                "init_noise_var"    : %(init_beta)f,
                "max_noise_var"  : 1,
                "learning_rate" : %(learning_rate)f,
                "min_misclass" : %(min_misclass)f,
                "max_misclass" : %(max_misclass)f,
                "noise_var_scale_up" : %(beta_scale_up)f,
                "noise_var_scale_down" : %(beta_scale_down)f,
                "time_constant" :  %(time_constant)f,
                "different_examples" : 1,
                "init_vis_prec" : 1,
                "learn_vis_prec" : 0
        },
    "algorithm": !obj:framework.training_algorithms.default.DefaultTrainingAlgorithm {
               "batch_size" : %(batch_size)d,
               "batches_per_iter" : 1000,
               "monitoring_batches" : 100,
               "monitoring_dataset" : !pkl: "/data/lisatmp/goodfeli/mnist_preprocessed_train.pkl"
    },
    "save_path": "%(save_path)s"
}
""" % locals()

    configfilepath = configsDir+"/config_"+str(i)+".yaml"

    f = open(configfilepath,'w')
    f.write(config)
    f.close()

    command += str(i)

""

command += '}}".yaml'
SkyNet.launch_job(command)

