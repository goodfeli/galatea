function cifar100_worker(features_path)
	len = numel(features_path);
	out_dir = features_path(1:(len-4));
	mkdir(out_dir)

	for Cexp = 1:5
		C = 10 ^ Cexp;
		cifar100_point_worker(C, features_path, out_dir)
	end
end
