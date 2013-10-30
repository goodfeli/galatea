CIFAR_DIR='/data/lisa/data/cifar10/cifar-10-batches-mat/';
SPAMS_DIR='/u/goodfeli/SPAMS/release/atlas64'; % E.g.: 'SPAMS/release/mkl64'

%%%%% Configuration
%addpath minFunc;
rfSize = 6;
numBases=1600;
CIFAR_DIM=[32 32 3];
alpha = 0.25;  %% CV-chosen value for soft-threshold function.
lambda = 1.0;  %% CV-chosen sparse coding penalty.

%%%%% Dictionary Training %%%%%%
alg='patches'; %% Use randomly sampled patches.  Test accuracy 79.14%
%alg='omp1';   %% Use 1-hot VQ (OMP-1).  Test accuracy 79.96%
%alg='sc';     %% Sparse coding

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

%% Check SPAMS install
if (strcmp(alg,'sc') || strcmp(encoder, 'sc'))
  assert(~strcmp(SPAMS_DIR, '/path/to/SPAMS/release/platform'), ...
         ['You need to modify sc_vq_demo.m so that SPAMS_DIR points to ' ...
          'the SPAMS toolkit release directory.  You can download this ' ...
          'toolkit from:  http://www.di.ens.fr/willow/SPAMS/downloads.html']);
  addpath(SPAMS_DIR);
end


%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);

fprintf('Preprocessing\n')
trainX = double([f1.data]);
clear f1;

num_patches = 10000;
dict_size = 6000;
patches = extract_patches(trainX, rfSize, CIFAR_DIM, num_patches + dict_size);

dict = patches(1:dict_size, :);
dict = bsxfun(@rdivide, dict, sqrt(sum(dict.^2,2)) + 1e-20);

patches = patches(dict_size+1:end, :);

fprintf('Running feature extraction\n')
% extract training features
tic
trainXC = time_extraction(patches, dict, 1.);
toc
