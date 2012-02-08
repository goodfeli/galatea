
function learn_cifar100_sc(lambda, save_path)
	CIFAR100_PATCHES='/data/lisa/data/cifar100/cifar100_patches/data.mat';

	%%%%% Configuration
	addpath minFunc;
	numBases=1600;

	D = load(CIFAR100_PATCHES);
	patches = double(D.patches);

	%patches = patches(1:100000,:);
	%fprintf(1,'warning: shrinking size of dataset\n')

	dictionary = minibatch_sc(patches, numBases, 10, lambda, 100000, .1);

	save(save_path, 'dictionary');
end
