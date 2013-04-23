jobdispatch --torque --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=72:00:00 --whitespace --gpu train.py $G/maxout/expdir/satmom_"{{3}}".yaml
