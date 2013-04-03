jobdispatch --torque --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=48:00:00 --whitespace --gpu train.py $G/boost/expdir/half_boost_"{{1,2,3,4,5,6}}".yaml
