import numpy as N
import SkyNet

job_name = "recons_srbm_mnist"

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

    nhid = 400
    irange = rng.uniform(1e-3,.05)
    beta = N.exp(rng.uniform(N.log(0.5),N.log(30.)))
    mean_field_iters = rng.randint(1,8)
    init_bias_hid = rng.uniform(-3.,0.0)
    damping_factor = rng.uniform(0,0.5)
    persistent_chains = rng.randint(1,20)
    enc_weight_decay = N.exp(rng.uniform(N.log(1e-6),N.log(3e-2)))
    learning_rate = N.exp(rng.uniform(N.log(1e-8),N.log(1e-3) ) )
    fold_biases = 0.
    batch_size = rng.randint(1,10)
    use_cd = rng.randint(0.5)
    save_path = modelsDir + '/model_'+str(i)+'.pkl'

    config = """
!obj:framework.scripts.train.Train {
    "dataset": !obj:framework.datasets.mnist.MNIST { "which_set" : "train" },
    "model": !obj:recons_srbm.br_recons_srbm.BR_ReconsSRBM {
                "nvis" : 784,
                "nhid" : %(nhid)d,
                "init_bias_hid" : %(init_bias_hid)f,
                "irange"  : %(irange)f,
                "beta"    : %(beta)f,
                "mean_field_iters" : %(mean_field_iters)d,
                "damping_factor" : %(damping_factor)f,
                "gibbs_iters" : 1,
                "persistent_chains" : %(persistent_chains)d,
                "enc_weight_decay" : %(enc_weight_decay)f,
                "learning_rate" : %(learning_rate)f,
                "fold_biases" : %(fold_biases)f,
                "use_cd" : %(use_cd)f
        },
    "algorithm": !obj:framework.training_algorithms.default.DefaultTrainingAlgorithm {
               "batch_size" : %(batch_size)d,
               "batches_per_iter" : 1000,
               "monitoring_batches" : 100,
               "monitoring_dataset" : !obj:framework.datasets.mnist.MNIST { "which_set" : "train" }
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

