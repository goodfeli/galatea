function valid_acc = cifar100_fold_point_worker(fold_str, C_str, features_path, out_dir)
	%this is like cifar100_fold_point_woker in the parent directory, but modified to work as a compiled standalone
	%and run on the cluster


	fold = str2num(fold_str)
	C = str2num(C_str)

	train_path = getenv('CIFAR100_TRAIN_PATH')

	fprintf(1,'loading cifar-100 just to get the labels\n')
	f = load(train_path);
	y = f.fine_labels + 1;
	clear f

	fprintf(1,'checking y\n')
	for i = 1:100
		if sum(y == i) <= 0
			i
			die die die
		end
	end
	
	fprintf(1,'loading features\n')
	X = load(features_path);
	try
		X = X.X;
	catch
		X = [ X.X_chunk_0; X.X_chunk_1 ];
	end
	fprintf(1,'augmenting features\n')
	X = [ X, ones(size(X,1),1) ];

	idx_mask = [ 1:50000 ] > (fold * 10000);
	idx_mask = idx_mask .* ( [1:50000 ] <= (fold+1) * 10000);

	train_X = double(X(~idx_mask,:));
	train_y = double(y(~idx_mask));
	

	fprintf(1,'checking train y\n')
	for i = 1:100
		if sum(y == i) <= 0
			fprintf(1,'%d\n',i)
			die die die
		end
	end


	if size(train_X,1) ~= 40000
		size(train_X,1)
		fold
		die die die
	end
	
	if size(train_y,1) ~= 40000
		die die die
	end

	valid_X = X(idx_mask == 1, :);
	valid_y = y(idx_mask == 1);

	if size(valid_X, 1) ~= 10000
		die die die
	end

	if size(valid_y, 1) ~= 10000
		die die die
	end


	clear X

	fprintf(1,'training svm\n')

	W = train_svm(train_X,train_y,C);

	cls = valid_X*W;
	mx = max(cls');
	mask = cls == repmat(mx',1,size(cls,2));
	pred_y = mask * [1:size(cls,2)]';

	valid_acc = mean(pred_y == valid_y);


	f = fopen( sprintf('%s/%d_%d.txt',out_dir,fold,C) ,'w');

	fprintf( f, '%d\n',valid_acc);

	fclose(f);
end
