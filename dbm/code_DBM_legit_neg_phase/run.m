randn('state',100);
rand('state',100);
warning off


fprintf(1,'Pretraining a Deep Boltzmann Machine. \n');
makebatches; 
[numcases numdims numbatches]=size(batchdata);


%%%%%% Training two-layer Boltzmann machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


