import numpy as N
import SkyNet

job_name = "recons_srbm_4000_1"

SkyNet.set_job_name(job_name)
configsDir = SkyNet.get_dir_path('configs')
modelsDir = SkyNet.get_dir_path('models')

rng = N.random.RandomState([1,2,3])

command = "--mem=2000 "+SkyNet.get_user_dir(tmp = False, force = SkyNet.cluster) + '/ift6266h11/framework/scripts/train.py '+configsDir+'/config_"{{'

first = True

num_jobs = 100

for i in xrange(num_jobs):

    if not first:
        command += ','
    ""

    first = False

    nhid = 4000
    irange = rng.uniform(.01,.03)
    if rng.uniform(0.,1.) > 0.5:
        learn_beta = 1
        beta_lr_scale = N.exp(rng.uniform(N.log(1e-5),N.log(1e-2)))
        beta = N.exp(rng.uniform(N.log(1.0),N.log(5)))
    else:
        learn_beta = 0
        beta_lr_scale = 1.
        beta = N.exp(rng.uniform(N.log(2),N.log(11)))

    mean_field_iters = rng.randint(5,13)
    no_damp_iters = rng.randint(0,1)

    init_bias_hid = rng.uniform(-3.,0.0)

    if rng.uniform(0.,1.) > 0.2:
        damping_factor = rng.uniform(0.4,0.5)
    else:
        damping_factor = rng.uniform(0, 0.1)

    if rng.uniform(0.,1.) > 0.2:
        persistent_chains = 1
        batch_size = rng.randint(1,10)
    else:
        persistent_chains = rng.randint(10,100)
        batch_size = persistent_chains
    enc_weight_decay = N.exp(rng.uniform(N.log(1e-6),N.log(3e-3)))
    learning_rate = N.exp(rng.uniform(N.log(1e-7),N.log(1e-3) ) )
    fold_biases = 0.
    use_cd = 0
    save_path = modelsDir + '/model_'+str(i)+'.pkl'

    config = """
!obj:framework.scripts.train.Train {
    "dataset": !pkl: &data "/data/lisatmp/goodfeli/cifar10_preprocessed_train_2M.pkl",
    "model": !obj:recons_srbm.br_recons_srbm.BR_ReconsSRBM {
                "nvis" : 192,
                "nhid" : %(nhid)d,
                "init_bias_hid" : %(init_bias_hid)f,
                "irange"  : %(irange)f,
                "init_beta"    : %(beta)f,
                "learn_beta" : %(learn_beta)d,
                "beta_lr_scale" : %(beta_lr_scale)f,
                "mean_field_iters" : %(mean_field_iters)d,
                "damping_factor" : %(damping_factor)f,
                "no_damp_iters" : %(no_damp_iters)d,
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
               "monitoring_batches" : 10,
               "monitoring_dataset" : *data
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

