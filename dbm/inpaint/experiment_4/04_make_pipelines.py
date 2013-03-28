num_jobs = 100

for i in xrange(num_jobs):
    path = '/RQexec/goodfell/experiment_4/%d/pipeline.sh' % i

    f = open(path, 'w')

    f.write("""
train.py /RQexec/goodfell/experiment_4/%(i)d/stage_00_inpaint.yaml
train.py /RQexec/goodfell/experiment_4/%(i)d/stage_01_validate.yaml
train.py /RQexec/goodfell/experiment_4/%(i)d/stage_02_final.yaml
""" % { 'i' : i })
    f.close()
