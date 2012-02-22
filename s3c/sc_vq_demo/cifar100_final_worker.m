function test_acc = cifar100_final_worker(C, features_path, test_features_path)

	addpath minFunc

	fprintf(1,'loading cifar-100 just to get the labels\n')
	f = load('/data/lisa/data/cifar100/cifar-100-matlab/train.mat');
	y = f.fine_labels + 1;
	clear f


	f = load('/data/lisa/data/cifar100/cifar-100-matlab/test.mat');
	test_y = f.fine_labels + 1;
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
	X = X.X;
	fprintf(1,'augmenting features\n')
	X = [ X, ones(size(X,1),1) ];

	testX = load(test_features_path);
	testX = testX.X;
	testX = [ testX, ones(size(X,1),1) ];

	fprintf(1,'checking train y\n')
	for i = 1:100
		if sum(y == i) <= 0
			fprintf(1,'%d\n',i)
			die die die
		end
	end


	if size(X,1) ~= 50000
		size(train_X,1)
		fold
		die die die
	end
	
	if size(y,1) ~= 50000
		die die die
	end

	valid_X = X(idx_mask == 1, :);
	valid_y = y(idx_mask == 1);

	fprintf(1,'training svm\n')

	W = train_svm(X,y,C);

	cls = test_X*W;
	mx = max(cls');
	mask = cls == repmat(mx',1,size(cls,2));
	pred_y = mask * [1:size(cls,2)]';

	test_acc = mean(pred_y == test_y);
end
