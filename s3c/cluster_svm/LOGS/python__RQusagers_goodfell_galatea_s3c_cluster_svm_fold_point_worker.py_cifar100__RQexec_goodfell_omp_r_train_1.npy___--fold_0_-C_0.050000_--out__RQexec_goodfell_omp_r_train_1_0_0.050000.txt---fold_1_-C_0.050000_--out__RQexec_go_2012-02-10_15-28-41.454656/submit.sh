#!/bin/sh

## Reasonable default values
# Execute the job from the current working directory.
#PBS -d /home/goodfell/galatea/s3c/cluster_svm

#All jobs must be submitted with an estimated run time
#PBS -l walltime=00:10:00

## Job name
#PBS -N dbi_74651eff401

## log out/err files
# We cannot use output_file and error_file here for now.
# We will use dbi_...out-id and dbi_...err-id instead
#PBS -o /home/goodfell/galatea/s3c/cluster_svm/LOGS/A
#PBS -e /home/goodfell/galatea/s3c/cluster_svm/LOGS/B

## Number of CPU (on the same node) per job
#PBS -l nodes=1:ppn=1

## Execute as many jobs as needed

#PBS -t 0-29

## Memory size (on the same node) per job
#PBS -l mem=18432mb

## Variable to put into the environment
#PBS -v OMP_NUM_THREADS=1,GOTO_NUM_THREADS=1,MKL_NUM_THREADS=1

## Execute the 'launcher' script in bash
# Bash is needed because we use its "array" data structure
# the -l flag means it will act like a login shell,
# and source the .profile, .bashrc, and so on
/bin/bash -l -e /home/goodfell/launcher
