randn('state',100);
rand('state',100);
warning off

clear all
close all

fprintf(1,'Converting Raw files into Matlab format \n');
converter; 


%%%%%% Training two-layer Boltzmann machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all 
numhid = 500; 
numpen = 1000;
maxepoch = 500;
  
fprintf(1,'Learning a Deep Bolztamnn Machine. \n');
restart=1;
makebatches; 
dbm_mf

%%%%%% Fine-tuning two-layer Boltzmann machine  for classification %%%%%%%%%%%%%%%%%
maxepoch=100;
makebatches; 
backprop


