base = """jobdispatch --torque --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=48:00:00 --whitespace --gpu train.py $G/dbm/inpaint/random_search_center_aux/exp/"{{%(args)s}}"/sup_center.yaml"""
args = ','.join([str(job_id) for job_id in xrange(25)])
f = open('launch.sh', 'w')
f.write(base % locals())
f.close()
