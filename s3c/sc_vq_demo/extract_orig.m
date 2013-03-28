function result = extract_orig(lambda, dict_path, feat_path)
	CIFAR_DIR='/data/lisa/data/cifar10/cifar-10-batches-mat/';
	SPAMS_DIR='/u/goodfeli/SPAMS/release/atlas64'; % E.g.: 'SPAMS/release/mkl64'

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
	encoder='sc'; encParam=lambda; %% Use sparse coding for encoder.


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
	f2=load([CIFAR_DIR '/data_batch_2.mat']);
	f3=load([CIFAR_DIR '/data_batch_3.mat']);
	f4=load([CIFAR_DIR '/data_batch_4.mat']);
	f5=load([CIFAR_DIR '/data_batch_5.mat']);

	trainX = double([f1.data; f2.data; f3.data; f4.data; f5.data]);
	clear f1 f2 f3 f4 f5;

	F = load('patches_and_prepro.mat');
	M = F.M;
	P = F.P;

	dictionary = load(dict_path)
	dictionary = dictionary.dictionary;

	% extract training features
	trainXC = extract_features(trainX, dictionary, rfSize, ...
				   CIFAR_DIM, M,P, encoder, encParam);
	%clear trainX;

	% standardize data
	trainXC_mean = mean(trainXC);
	trainXC_sd = sqrt(var(trainXC)+0.01);
	trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
	%clear trainXC;
	trainXCs = [trainXCs, ones(size(trainXCs,1),1)]; % intercept term


	save(feat_path,'trainXCs','-v7.3');
end
