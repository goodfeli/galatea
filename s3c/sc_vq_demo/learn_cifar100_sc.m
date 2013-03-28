
function learn_cifar100_sc(lambda, save_path, numBases, iters, batch_size, new_coeff)

	if nargin < 3
		numBases = 1600
	end

	if nargin < 4
		iters = 10
	end

	if nargin < 5
		batch_size = 100000
	end

	if nargin < 6
		new_coeff = .1
	end

	CIFAR100_PATCHES='/data/lisa/data/cifar100/cifar100_patches/data.mat';

	%%%%% Configuration

	D = load(CIFAR100_PATCHES);
	patches = double(D.patches);

	%patches = patches(1:100000,:);
	%fprintf(1,'warning: shrinking size of dataset\n')

	dictionary = minibatch_sc(patches, numBases, iters, lambda, batch_size, new_coeff);

	save(save_path, 'dictionary');
end
