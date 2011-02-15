#loose translation into python of samplecode/main.m
import sys
import time
import serialutil
import quantize_data
import learning_curve_dataset
import save_data
import make_zip
import make_learning_curve
import alc

#For now, we just hardcode our preferences in here

#1) Root directory-- this determines where your fils will go

dir_root = '.'

#CLOP objects not supported in the python translation, as far as I can tell from
#briefly checking the docs they are a Matlab thing
dir_prepro  = dir_root +  '/Prepro'    # Directory of the resulting preprocessed data
dir_zip     = dir_root + '/Zipped'    # ** Zipped files ready to go!! ** (zips of prepro)
dir_resu    = dir_root + '/Results'   # Directory of the results (only for toy data)

# 2) Identify your experiment
# ----------------------------
# The nickname you used on the challenge website
# (it does not matter if you choose something else here)
my_workbench_id = 'myname';

# The date
the_date        = time.strftime("%y_%d_%m_%H_%M_%S",time.localtime())

# 2) Setup of your experiment
# ----------------------------
# The list of datasets to work on:
datasets = ['ule' ]
#possibilties are 'ule', 'avicenna', 'harry', 'rita', 'sylvester', 'terry'


# Name of preprocessing algorithm to use (for now, only 'raw' is supported, we probably
# want to make up a different interface for specifying the algorithm)
prepro_algos = [ 'raw' ]

# The list of data subsets to work on (during development, omitting the
# final evaluation set, saves time and creates smaller submissions).
set_names=['devel', 'valid', 'final']

# Choose whether to zip only the validation data if the files are big
zip_final_only_if_small=1;

# -o-|-o-|-o-|-o-|-o-|-o-|-o- END USER-PREFERENCES -o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-

#%% Initializations
serialutil.mkdir(dir_resu)
serialutil.mkdir(dir_prepro)
serialutil.mkdir(dir_zip)
debug=0

# Learning curve setting used for evaluation:
min_repeat=10    # min. num. repeats for each point on learning curve
max_repeat=500   # max. num. repeats for each point on learning curve
ebar=0.01        # max. desired error bar
max_point_num=7  # maximum number of points on the learning curve

# LOOP OVER DATASETS
# ===================
for k, data_name in enumerate(datasets):


    print '\n-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-'
    print '\n-o-|-o-|-o-|-o-|-o-|-o-      '+data_name+'      -o-|-o-|-o-|-o-|-o-|-o-'
    print '\n-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-|-o-\n'

    #Load the data
    try:
        #print "TODO: make X0, Y, p from the current dataset (use pylearn) "
        print "WARNING: right now data is not normalized at all, add an option for this"
        dataset = learning_curve_dataset.learning_curve_dataset(data_name)
    except Exception, e:
        print "Unabled to open "+data_name
        print "Due to the following exception: "
        print e
        print "(Continuing to next dataset)"
        print ""
        continue

    # LOOP OVER ALGORITHMS
    # =====================
    for h, palg in enumerate(prepro_algos):

        print '\n-o-|-o-|-o-|-o-|-o-|-o-      '+palg+'      -o-|-o-|-o-|-o-|-o-|-o-'
        prepro_name  = my_workbench_id + '_' +  data_name + '_' + palg + '_' + the_date

        if palg ==  'raw':
            proc_data = dataset;
        else:
            print "TODO: support running (and timing) of custom preprop algorithms"
        #endif

        # If your number of features n exceeds the number of examples m,
        # replace your data matrices X (m, n) by the smaller matrix X*X'
        # of dimension (m, m) -- we'll notice automatically:

        if proc_data.X0['valid'].shape[1] > proc_data.X0['valid'].shape[0]:
            proc_data = kernelize(proc_data);
        #endif

        # Optionally, quantize data (precision is usually unnecessary and this makes the submissions a lot smaller)
        proc_data = quantize_data.quantize_data(proc_data)

        # Save the data (note: the name convention is flexible but files must
        # contain the dataset name and end by _valid.prepro or _final.prepro.
        save_data.save_data(dir_prepro, prepro_name, proc_data)

        # Create a zip file with the query in the Queries directory
        make_zip.make_zip(dir_prepro, dir_zip, prepro_name, zip_final_only_if_small)

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%%%%%%% The zip files must be submitted to the web site %%%%%%
        #%%%%%%% IMPORTANT: do not submit preprocessed data for  %%%%%%
        #%%%%%%% the development set (too big and not evaluated).%%%%%%
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # For the toy example, you can run the scoring software yourself:
        snames = sorted([set_name for set_name in set_names if set_name != 'devel'])
        for i, sname in enumerate(snames):

            t1 = time.time()
            #Get the truth values (labels)
            Y = dataset.Y

            if Y == None:
                print "Couldn't get labels for "+sname
                print "Continuing to next dataset"
                print ""
            #endif

            full_resu_name= prepro_name +  '_' + sname
            resu_name= data_name + '_' + palg + '_' + sname
            print '==> Making '+resu_name+' learning curve'

            #Make the learning curve
            x, y, e  = make_learning_curve.make_learning_curve( proc_data.X0[sname], Y[sname], min_repeat, max_repeat, ebar, max_point_num, useRPMat=True)

            # Compute the (normalized) area under the learning curve
            score=alc.alc(x, y)
            print '==> Normalized ALC for '+resu_name+'  = '+str(score)

            t2 = time.time()

            difftime = t2 - t1

            print '==> Learning curve completed in '+str(difftime)+' seconds'

            # Plot the learning curve
            print "TODO: implement plotting of learning curve / saving of results"
            """hh = plot_learning_curve(resu_name, score, x, y, e)

            # Save the results
            save(dir_resu + '/' + full_resu_name + '.txt', 'x', 'y', 'e', '-ascii' ) #todo-- can this be done with scipy.io.savemat?
            saveas(hh, dir_resu + '/' + full_resu_name + '.png' )"""
    # end for loop over preprocessings
# end for loop over datasets

#%%%%%%%%%%%%%%%%%%%%% END MAIN %%%%%%%%%%%%%%%%%%%%%%%
