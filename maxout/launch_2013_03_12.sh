jobdispatch --torque --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=48:00:00 --whitespace --gpu train.py $G/maxout/expdir/cifar10_3C7C"{{S,T,U}}".yaml
