makebatches
[numcases numdims numbatches]=size(batchdata);
data = [batchdata(:,:,1)];
N=1; 

load fullmnist_dbm
D = 5;
N1 = 6;
N2 = 7;
ofs = 622;
data = data(1,ofs:(ofs+D-1));

vishid =  10 * vishid(ofs:(ofs+D-1),1:N1);
hidpen =  10 * hidpen(1:N1,1:N2);
visbiases = visbiases(1,ofs:(ofs+D-1));
hidbiases = hidbiases(1,1:N1);
penbiases = penbiases(1,1:N2);

[numdims numhids] = size(vishid);
[numhids numpens] = size(hidpen); 

[numcases numdims numbatches]=size(batchdata);
N=numcases;
[h1, h2] = ...
     mf_class(data,vishid,hidbiases,visbiases,hidpen,penbiases);


save('model.mat','vishid','hidpen','visbiases','hidbiases','penbiases') 
save('data.mat', 'data', 'h1', 'h2' )
