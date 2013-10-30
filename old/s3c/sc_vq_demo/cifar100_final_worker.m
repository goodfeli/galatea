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
	try
		X = X.X;
	catch
		X = [X.X_chunk_0; X.X_chunk_1;...
X.X_chunk_2; X.X_chunk_3; X.X_chunk_4; X.X_chunk_5; X.X_chunk_6;...
X.X_chunk_7; X.X_chunk_8; X.X_chunk_9];
	end
	X = double(X);
	fprintf(1,'augmenting features\n')
	X = [ X, ones(size(X,1),1) ];

	testX = load(test_features_path);
	try
		testX = testX.X;
	catch
		testX = [testX.X_chunk_0; testX.X_chunk_1];
	end
	testX = [ testX, ones(size(testX,1),1) ];

	fprintf(1,'checking train y\n')
	for i = 1:100
		if sum(y == i) <= 0
			fprintf(1,'%d\n',i)
			die die die
		end
	end


	if size(X,1) ~= 50000
		size(X,1)
		size(X)
		die die die
	end
	
	if size(y,1) ~= 50000
		die die die
	end

	fprintf(1,'training svm\n')

	W = train_svm(X,y,C);

	cls = testX*W;
	mx = max(cls');
	mask = cls == repmat(mx',1,size(cls,2));
	pred_y = mask * [1:size(cls,2)]';

	test_acc = mean(pred_y == test_y);
end
