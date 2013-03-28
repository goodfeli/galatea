function [f, df] = ECG1(VV,Dim,XX,target,temp_h2);

numdims = Dim(1); 
numhids = Dim(2);
numpens = Dim(3); 
N = size(XX,1);

X=VV;
% Do decomversion.
 w1_vishid = reshape(X(1:numdims*numhids),numdims,numhids);
 xxx = numdims*numhids;
 w1_penhid = reshape(X(xxx+1:xxx+numpens*numhids),numpens,numhids);
 xxx = xxx+numpens*numhids;
 hidpen = reshape(X(xxx+1:xxx+numhids*numpens),numhids,numpens);
 xxx = xxx+numhids*numpens;
 w_class = reshape(X(xxx+1:xxx+numpens*10),numpens,10);
 xxx = xxx+numpens*10;
 hidbiases = reshape(X(xxx+1:xxx+numhids),1,numhids);
 xxx = xxx+numhids;
 penbiases = reshape(X(xxx+1:xxx+numpens),1,numpens);
 xxx = xxx+numpens;
 topbiases = reshape(X(xxx+1:xxx+10),1,10);
 xxx = xxx+10;

  bias_hid= repmat(hidbiases,N,1);
  bias_pen = repmat(penbiases,N,1);
  bias_top = repmat(topbiases,N,1);

  w1probs = 1./(1 + exp(-XX*w1_vishid -temp_h2*w1_penhid - bias_hid  ));
  w2probs = 1./(1 + exp(-w1probs*hidpen - bias_pen));
  targetout = exp(w2probs*w_class + bias_top );
  targetout = targetout./repmat(sum(targetout,2),1,10);

  f = -sum(sum( target(:,1:end).*log(targetout)));

  IO = (targetout-target(:,1:end));
  Ix_class=IO; 
  dw_class =  w2probs'*Ix_class;
  dtopbiases = sum(Ix_class);

 Ix2 = (Ix_class*w_class').*w2probs.*(1-w2probs);
 dw2_hidpen =  w1probs'*Ix2;
 dw2_biases = sum(Ix2); 

 Ix1 = (Ix2*hidpen').*w1probs.*(1-w1probs); 
 dw1_penhid =  temp_h2'*Ix1;
 dw1_vishid = XX'*Ix1;
 dw1_biases = sum(Ix1);

 dhidpen = 0*dw2_hidpen;
 dw1_penhid = 0*dw1_penhid;
 dw1_vishid = 0*dw1_vishid;
 dw2_biases = 0*dw2_biases;
 dw1_biases = 0*dw1_biases; 

 df = [dw1_vishid(:)' dw1_penhid(:)' dw2_hidpen(:)' dw_class(:)' dw1_biases(:)' dw2_biases(:)' dtopbiases(:)']'; 


