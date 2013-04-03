                #!/bin/sh

                ## Reasonable default values
                # Execute the job from the current working directory.
                #PBS -d /exec5/GROUP/lisa/goodfell/galatea/maxout

                #All jobs must be submitted with an estimated run time
                #PBS -l walltime=48:00:00

                ## Job name
                #PBS -N dbi_b4d550aa5d0

                ## log out/err files
                # We cannot use output_file and error_file here for now.
                # We will use dbi_...out-id and dbi_...err-id instead
                #PBS -o /exec5/GROUP/lisa/goodfell/galatea/maxout/LOGS/train.py__RQexec_goodfell_galatea_boost_expdir_half_boost___1-2-3-4-5-6__.yaml_2013-03-25_19-18-19.900310/dbi_b4d550aa5d0.out
                #PBS -e /exec5/GROUP/lisa/goodfell/galatea/maxout/LOGS/train.py__RQexec_goodfell_galatea_boost_expdir_half_boost___1-2-3-4-5-6__.yaml_2013-03-25_19-18-19.900310/dbi_b4d550aa5d0.err


                ## Number of CPU (on the same node) per job
                #PBS -l nodes=1:ppn=1

                ## Execute as many jobs as needed
                #PBS -t 0-5

                ## Queue name
                #PBS -q @hades
export THEANO_FLAGS=device=gpu,floatX=float32,force_device=True
export OMP_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1

                ## Variable to put into the environment
                #PBS -v THEANO_FLAGS,OMP_NUM_THREADS,GOTO_NUM_THREADS,MKL_NUM_THREADS

                ## Execute the 'launcher' script in bash
                # Bash is needed because we use its "array" data structure
                # the -l flag means it will act like a login shell,
                # and source the .profile, .bashrc, and so on
                /bin/bash -l -e /exec5/GROUP/lisa/goodfell/galatea/maxout/LOGS/train.py__RQexec_goodfell_galatea_boost_expdir_half_boost___1-2-3-4-5-6__.yaml_2013-03-25_19-18-19.900310/launcher
