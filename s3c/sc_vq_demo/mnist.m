addpath minFunc;
use_g = 0;
normalize = 0;
C = 1e3

load ../../pddbm/pddbm_train_features.mat
if use_g
	X = [ X , G ];
	assert(size(X,1) == 60000)
	assert(size(X,2) == 784 + size(G,2) )
end

y = y + 1;


%normalize features
if normalize
	mn = mean(X);
	sd = sqrt(.01 + var(X));
	X = X- repmat(mn,size(X,1),1);
	X = X ./ repmat(sd,size(X,1),1);
end

trainX = X(1:50000,:);
trainY = y(1:50000);
validX = X(50001:60000,:);
validY = y(50001:60000);

theta = train_svm_hardcore(trainX, trainY, C);

[val,labels] = max(trainX*theta, [], 2);
subtrain = 100 * (1 - sum(labels ~= trainY) / length(trainY));
fprintf('Train (50k) accuracy %f%%\n', subtrain);

[val,labels] = max(validX*theta, [], 2);
valid = 100 * (1 - sum(labels ~= validY) / length(validY));
fprintf('Valid accuracy %f%%\n', valid);

theta = train_svm_hardcore(X, y, C);
[val,labels] = max(X*theta, [], 2);
train = 100 * (1 - sum(labels ~= y) / length(y));
fprintf('Train (full) accuracy %f%%\n', train);

clear X
clear y

load ../../pddbm/pddbm_test_features.mat
if use_g
	X = [ X , G ];
	assert(size(X,1) == 10000)
	assert(size(X,2) == 784 + size(G,2) )
end
if normalize
	X = X- repmat(mn,size(X,1),1);
	X = X ./ repmat(sd,size(X,1),1);
end
y = y + 1;
[val,labels] = max(X*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= y) / length(y)));

subtrain
valid
train
