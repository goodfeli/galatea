
function learn_cifar100_omp(save_path)
	CIFAR100_PATCHES='/data/lisa/data/cifar100/cifar100_patches/data.mat';

	%%%%% Configuration
	numBases=1600;

	D = load(CIFAR100_PATCHES);
	patches = double(D.patches);

	dictionary = run_omp1(patches, numBases, 20);

	save(save_path, 'dictionary');
end
