function cifar100_point_worker(C, features_path, out_dir)

	for fold=0:4
		cifar100_fold_point_worker(fold, C, features_path, out_dir)
	end
end
