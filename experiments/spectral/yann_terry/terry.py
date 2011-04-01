import numpy
import theano
import nnet
import gzip
import time
import scipy.sparse
import zipfile
from tempfile import TemporaryFile
from jobman import DD
import jobman, jobman.sql
import subprocess as sub


numpy.random.seed(0x7265257d5f)


def create_submission(encode, epoch, valid, test):
    encoded_valid = encode(valid)
    encoded_test = encode(test)
    
    valid_file = TemporaryFile()
    test_file = TemporaryFile()
    
    numpy.savetxt(valid_file, encoded_valid, fmt="%.3f")
    numpy.savetxt(test_file, encoded_test, fmt="%.3f")
    
    valid_file.seek(0)
    test_file.seek(0)
    
    submission = zipfile.ZipFile("terry_sda%d.zip" % epoch, "w", compression=zipfile.ZIP_DEFLATED)
    
    submission.writestr("terry_sda_valid.prepro", valid_file.read())
    submission.writestr("terry_sda_final.prepro", test_file.read())
    
    submission.close()
    valid_file.close()
    test_file.close()


def main(layers=[('L', 1000),('L', 500),],
		 learning_rate=0.1,
		 batch_size=10,
		 epochs=50,
		 corruption_levels=[0.2, 0.5],
		 sampling=0.005,
		 state=None,
		 channel=None,
		 **kwargs):
    print "Loading dataset..."
    train = scipy.sparse.csr_matrix(numpy.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_train.npy.gz")), dtype=theano.config.floatX)
    valid = scipy.sparse.csr_matrix(numpy.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_valid.npy.gz")), dtype=theano.config.floatX)[1:]
    test = scipy.sparse.csr_matrix(numpy.load(gzip.open("/data/lisa/data/UTLC/sparse/terry_test.npy.gz")), dtype=theano.config.floatX)[1:]
    
    train.data = numpy.sign(train.data)
    valid.data = numpy.sign(valid.data)
    test.data = numpy.sign(test.data)
    
    train_inds = range(train.shape[0])
    numpy.random.shuffle(train_inds)
    
    print "Building model..."
    if type(layers) is str:
        layers = eval(layers)
    
    if type(corruption_levels) is str:
        corruption_levels = eval(corruption_levels)
    
    model = nnet.MLP(n_in=train.shape[1],
                     layers=layers,
				     learning_rate=learning_rate,
				     corruption_levels=corruption_levels,
				     sampling=sampling)

    n_train_batches = len(train_inds) / batch_size

    n_validation_period =  n_train_batches / 100
    
    epoch_times = []
    
    print "Training model..."
    for layer in range(len(layers)):
        for epoch in range(epochs):
            begin = time.time()
        
            losses = []
            for minibatch in range(n_train_batches):
                loss = model.pretrain(train[train_inds[minibatch::n_train_batches]], layer)
                
                losses.append(loss)
            end = time.time()
            
            epoch_time = (end - begin) / 60
            
            loss = numpy.mean(losses)
            
            epoch_times.append(epoch_time)
            
            mean_epoch_time = numpy.mean(epoch_times)
            
            create_submission(model.outputp, epoch, valid, test)
            
            if channel != None:
                state.current_layer = layer
                state.epoch = epoch
                state.epoch_time = mean_epoch_time
                state.loss = loss
                
                channel.save()
            
            print "layer = %d, epoch = %d, time = %.2f, mean_time = %.2f, loss = %.4f" % (layer, epoch, epoch_time, mean_epoch_time, loss)
    
	model.save()
	
    if channel != None:
        state.epoch = epoch
        state.epoch_time = mean_epoch_time
        state.loss = loss
    
    return


def jobman_entrypoint(state, channel):
    main(state=state, channel=channel, **state)

    return channel.COMPLETE


def produit_cartesien_jobs(val_dict):
    job_list = [DD()]
    all_keys = val_dict.keys()

    for key in all_keys:
        possible_values = val_dict[key]
        new_job_list = []
        for val in possible_values:
            for job in job_list:
                to_insert = job.copy()
                to_insert.update({key: val})
                new_job_list.append(to_insert)
        job_list = new_job_list

    return job_list


def jobman_insert():
    # jobdispatch --condor --repeat_jobs=24 jobman sql -n 1 'postgres://dauphiya:wt17se79@gershwin/dauphiya_db/terry' .
    JOBDB = 'postgres://dauphiya:wt17se79@gershwin/dauphiya_db/terry'
    EXPERIMENT_PATH = "terry.jobman_entrypoint"
    JOB_VALS = {
        'layers' : [str([('L', 1000), ('L', i)]) for i in 100, 500, 1000, 2000],
        'learning_rate' : [0.1],
        'corruption_levels' : [str([0.5, 0.1]), str([0.5, 0.2]), str([0.5, 0.3])],
        'exp_tag' : ['sda'],
        }

    jobs = produit_cartesien_jobs(JOB_VALS)
    
    answer = raw_input("Submit %d jobs?[y/N] " % len(jobs))
    if answer == "y":
        numpy.random.shuffle(jobs)

        db = jobman.sql.db(JOBDB)
        for job in jobs:
            print job
            job.update({jobman.sql.EXPERIMENT: EXPERIMENT_PATH})
            jobman.sql.insert_dict(job, db)

        print "inserted %d jobs" % len(jobs)

def jobman_view():
    URL = 'postgres://dauphiya:wt17se79@gershwin/dauphiya_db/'
    TABLE_NAME = 'terry'
    vname = TABLE_NAME + "_view"
    view_proc = sub.Popen(("jobman sqlview "+URL+TABLE_NAME+" "+vname).split(),stdout=sub.PIPE)
    view_return = view_proc.wait()
    print "View returned",view_return
    proc = sub.Popen("psql -h gershwin -d dauphiya_db -U dauphiya ".split(),stdin=sub.PIPE,stdout=sub.PIPE)
    cjob = ', layers, learningrate, corruptionlevels, corruptionlevels_s, exptag, jobman_status, epoch, epochtime, loss'
    request = "select id"+cjob+" from "+vname+''' order by loss DESC;\n'''
    print proc.communicate(request)[0]
    print request

if __name__ == "__main__":
    import sys
    if "insert" in sys.argv:
        jobman_insert()
    elif "view" in sys.argv:
        jobman_view()
    else:
        main()

