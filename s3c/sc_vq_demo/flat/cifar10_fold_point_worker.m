function valid_acc = cifar10_fold_point_worker(fold_str, C_str, features_path, out_dir)
	%this is like cifar100_fold_point_worker but modified to run on cifar10

	fold = str2num(fold_str)
	C = str2num(C_str)

	data_dir = getenv('CIFAR10_PATH')

	fprintf(1,'loading cifar-10 just to get the labels\n')
	CIFAR_DIR = data_dir;
	f1=load([CIFAR_DIR '/data_batch_1.mat']);
	f2=load([CIFAR_DIR '/data_batch_2.mat']);
	f3=load([CIFAR_DIR '/data_batch_3.mat']);
	f4=load([CIFAR_DIR '/data_batch_4.mat']);
	f5=load([CIFAR_DIR '/data_batch_5.mat']);

	y = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!
	clear f1 f2 f3 f4 f5;

	fprintf(1,'checking y\n')
	for i = 1:10
		if sum(y == i) <= 0
			i
			die die die
		end
	end
	
	fprintf(1,'loading features\n')
	X = load(features_path);
	X = X.trainXCs;

	idx_mask = [ 1:50000 ] > (fold * 10000);
	idx_mask = idx_mask .* ( [1:50000 ] <= (fold+1) * 10000);

	train_X = double(X(~idx_mask,:));
	train_y = double(y(~idx_mask));
	

	fprintf(1,'checking train y\n')
	for i = 1:10
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

	fprintf(1,'done training svm\n')

	cls = valid_X*W;
	mx = max(cls');
	mask = cls == repmat(mx',1,size(cls,2));
	pred_y = mask * [1:size(cls,2)]';

	valid_acc = mean(pred_y == valid_y);


	f = fopen( sprintf('%s/%d_%d.txt',out_dir,fold,C) ,'w');

	fprintf( f, '%d\n',valid_acc);

	fclose(f);
end
