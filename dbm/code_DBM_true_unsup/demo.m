randn('state',100);
rand('state',100);
warning off

clear all
close all

fprintf(1,'Converting Raw files into Matlab format \n');
converter; 

fprintf(1,'Pretraining a Deep Boltzmann Machine. \n');
makebatches; 
[numcases numdims numbatches]=size(batchdata);

%%%%%% Training 1st layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numhid=500; maxepoch=100;
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
rbm

%%%%%% Training 2st layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all 
numpen = 1000; 
maxepoch=200;
fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
restart=1;
makebatches; 
rbm_l2


%%%%%% Training two-layer Boltzmann machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all 
numhid = 500; 
numpen = 1000;
maxepoch=300; %To get results in the paper I used maxepoch=500, which took over 2 days or so. 
  
fprintf(1,'Learning a Deep Bolztamnn Machine. \n');
restart=1;
makebatches; 
dbm_mf

%%%%%% Fine-tuning two-layer Boltzmann machine  for classification %%%%%%%%%%%%%%%%%
maxepoch=100;
makebatches; 
backprop


