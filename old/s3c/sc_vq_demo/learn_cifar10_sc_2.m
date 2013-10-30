function result = learn_cifar10_sc_2(lambda, save_path)
	CIFAR_DIR='/data/lisa/data/cifar10/cifar-10-batches-mat/';
	SPAMS_DIR='/u/goodfeli/SPAMS/release/atlas64'; % E.g.: 'SPAMS/release/mkl64'

	%%%%% Configuration
	rfSize = 6;
	numBases=1600;
	CIFAR_DIM=[32 32 3];

	alg='sc';     %% Sparse coding


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


	F = load('patches_and_prepro.mat');
	patches = F.patches;

	% run training
	dictionary = run_sc(patches, numBases, 10, lambda);

	save(save_path, 'dictionary')
end
