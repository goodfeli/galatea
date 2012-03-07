#!/bin/sh

## Reasonable default values
# Execute the job from the current working directory.
#PBS -d /home/goodfell/galatea/s3c/cluster_svm

#All jobs must be submitted with an estimated run time
#PBS -l walltime=48:00:00

## Job name
#PBS -N dbi_552e29b9981

## log out/err files
# We cannot use output_file and error_file here for now.
# We will use dbi_...out-id and dbi_...err-id instead
#PBS -o /home/goodfell/galatea/s3c/cluster_svm/LOGS/cifar100_fold_point_worker___0_10.000000-1_10.000000-2_10.000000-3_10.000000-4_10.000000-0_100.000000-1_100.000000-2_100.000000-3_100.000000-4_100.000000-0_1000.000000-1_1000.000000-2_1000.000000-3_1000.000000-4_1000.000000-0_10_2012-02-22_16-54-17.843785/dbi_552e29b9981.out
#PBS -e /home/goodfell/galatea/s3c/cluster_svm/LOGS/cifar100_fold_point_worker___0_10.000000-1_10.000000-2_10.000000-3_10.000000-4_10.000000-0_100.000000-1_100.000000-2_100.000000-3_100.000000-4_100.000000-0_1000.000000-1_1000.000000-2_1000.000000-3_1000.000000-4_1000.000000-0_10_2012-02-22_16-54-17.843785/dbi_552e29b9981.err

## Number of CPU (on the same node) per job
#PBS -l nodes=1:ppn=1

## Execute as many jobs as needed

#PBS -t 0-24

## Memory size (on the same node) per job
#PBS -l mem=12288mb

## Variable to put into the environment
#PBS -v OMP_NUM_THREADS=1,GOTO_NUM_THREADS=1,MKL_NUM_THREADS=1

## Execute the 'launcher' script in bash
# Bash is needed because we use its "array" data structure
# the -l flag means it will act like a login shell,
# and source the .profile, .bashrc, and so on
/bin/bash -l -e /home/goodfell/galatea/s3c/cluster_svm/LOGS/cifar100_fold_point_worker___0_10.000000-1_10.000000-2_10.000000-3_10.000000-4_10.000000-0_100.000000-1_100.000000-2_100.000000-3_100.000000-4_100.000000-0_1000.000000-1_1000.000000-2_1000.000000-3_1000.000000-4_1000.000000-0_10_2012-02-22_16-54-17.843785/launcher
