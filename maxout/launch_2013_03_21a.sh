jobdispatch --torque --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=60:00:00 --whitespace --gpu train.py $G/maxout/expdir/david_full_wd2.yaml
