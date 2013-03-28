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


function  [temp_h1, temp_h2] = ...
   mf(data,targets,vishid,hidbiases,visbiases,hidpen,penbiases,labpen,hidrecbiases);

[numdim numhid]=size(vishid);
[numhid numpen]=size(hidpen);

  numcases = size(data,1);
  bias_hid= repmat(hidbiases,numcases,1);
  bias_pen = repmat(penbiases,numcases,1);
  big_bias =  data*vishid; 
  lab_bias =  targets*labpen; 

 temp_h1 = 1./(1 + exp(-data*(2*vishid) - repmat(hidbiases,numcases,1)));
 temp_h2 = 1./(1 + exp(-temp_h1*hidpen - targets*labpen - bias_pen));

 temp_h1_old = temp_h1;
 temp_h2_old = temp_h2;

 for ii= 1:10 % Number of the mean-field updates. I also used 30 MF updates.  
   totin_h1 = big_bias + bias_hid + (temp_h2*hidpen');
   temp_h1_new = 1./(1 + exp(-totin_h1));

   totin_h2 =  (temp_h1_new*hidpen + bias_pen + lab_bias);
   temp_h2_new = 1./(1 + exp(-totin_h2));

  diff_h1 = sum(sum(abs(temp_h1_new - temp_h1),2))/(numcases*numhid);
  diff_h2 = sum(sum(abs(temp_h2_new - temp_h2),2))/(numcases*numpen);
   fprintf(1,'\t\t\t\tii=%d Mean-Field: h1=%f h2=%f\r',ii,diff_h1,diff_h2);
    if (diff_h1 < 0.0000001 & diff_h2 < 0.0000001)
      break;
    end
   temp_h1 = temp_h1_new;
   temp_h2 = temp_h2_new;
 end

 temp_h1 = temp_h1_new;
 temp_h2 = temp_h2_new;



