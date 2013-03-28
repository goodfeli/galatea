% Version 1.000 
%
% Code provided by Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

if restart ==1,
  restart=0;

epsilonw      = 0.05;   % Learning rate for weights 
epsilonvb     = 0.05;   % Learning rate for biases of visible units 
epsilonhb     = 0.05;   % Learning rate for biases of hidden units 

CD=1;   
weightcost  = 0.001;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases numdims numbatches]=size(batchdata);
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.001*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end


for epoch = epoch:maxepoch
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,
 fprintf(1,'epoch %d batch %d\r',epoch,batch); 

 visbias = repmat(visbiases,numcases,1);
 hidbias = repmat(2*hidbiases,numcases,1); 
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  data = data > rand(numcases,numdims);  

  poshidprobs = 1./(1 + exp(-data*(2*vishid) - hidbias));    
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);
  negdata = 1./(1 + exp(-poshidstates*vishid' - visbias));
  negdata = negdata > rand(numcases,numdims); 
  neghidprobs = 1./(1 + exp(-negdata*(2*vishid) - hidbias));

  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));
  errsum = err + errsum;

   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;
%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   if rem(batch,600)==0  
     figure(1); 
     dispims(negdata',28,28);
     drawnow
   end  
  end
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 

end;

save fullmnistvh vishid visbiases hidbiases epoch 


