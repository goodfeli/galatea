CIFAR_DIR='/data/lisa/data/cifar10/cifar-10-batches-mat/';

%%%%% Configuration
addpath minFunc;
rfSize = 6;
numBases=1600;
CIFAR_DIM=[32 32 3];
alpha = 0.25;  %% CV-chosen value for soft-threshold function.
lambda = 1.0;  %% CV-chosen sparse coding penalty.

%%%%% Dictionary Training %%%%%%
%alg='patches'; %% Use randomly sampled patches.  Test accuracy 79.14%
%alg='omp1';   %% Use 1-hot VQ (OMP-1).  Test accuracy 79.96%
alg='sc';     %% Sparse coding

%%%%% Encoding %%%%%%
%encoder='thresh'; encParam=alpha; %% Use soft threshold encoder.
encoder='sc'; encParam=lambda; %% Use sparse coding for encoder.

%%%%% SVM Parameter %%%%%
switch (encoder)
 case 'thresh'
  L = 0.01; % L=0.01 for 1600 features.  Use L=0.03 for 4000-6000 features.
 case 'sc'
  L = 1.0; % May need adjustment for various combinations of training / encoding parameters.
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check CIFAR directory
assert(~strcmp(CIFAR_DIR, '/path/to/cifar/cifar-10-batches-mat/'), ...
       ['You need to modify sc_vq_demo.m so that CIFAR_DIR points to ' ...
        'your cifar-10-batches-mat directory.  You can download this ' ...
        'data from:  http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz']);

%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);

trainX = double([f1.data; f2.data; f3.data; f4.data; f5.data]);
trainY = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!
clear f1 f2 f3 f4 f5;

% extract random patches
switch (alg)
 case 'omp1'
  numPatches = 400000;
 case 'sc'
  numPatches = 100000;
 case 'patches'
  numPatches = 50000; % still needed for whitening
end
patches = zeros(numPatches, rfSize*rfSize*3);
for i=1:numPatches
  if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);
  patch = reshape(trainX(random('unid', size(trainX,1)),:), CIFAR_DIM);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
  patches(i,:) = patch(:)';
end

% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% ZCA whitening (with low-pass)
C = cov(patches);
M = mean(patches);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
patches = bsxfun(@minus, patches, M) * P;

dictionary = run_sc(patches, numBases, 10, lambda);

save('/u/goodfeli/test_sc.mat','dictionary')
