function test_acc = cifar10_final_worker(C, train_features_path, test_features_path)

	addpath minFunc

	CIFAR_DIR='/data/lisa/data/cifar10/cifar-10-batches-mat/';
	f1=load([CIFAR_DIR '/data_batch_1.mat']);
	f2=load([CIFAR_DIR '/data_batch_2.mat']);
	f3=load([CIFAR_DIR '/data_batch_3.mat']);
	f4=load([CIFAR_DIR '/data_batch_4.mat']);
	f5=load([CIFAR_DIR '/data_batch_5.mat']);

	fprintf(1,'loading cifar-10 just to get the labels\n')
	f = load('/data/lisa/data/cifar100/cifar-100-matlab/train.mat');
	y = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!
	clear f

	fprintf(1,'checking y\n')
	for i = 1:10
		if sum(y == i) <= 0
			i
			die die die
		end
	end
	
	fprintf(1,'loading features\n')
	X = load(train_features_path);
	try
		X = X.X;
		fprintf(1,'augmenting features\n')
		X = [ X, ones(size(X,1),1) ];
	catch
		X = X.trainXCs;
	end


	X = double(X);
	y = double(y);
	

	fprintf(1,'checking train y\n')
	for i = 1:10
		if sum(y == i) <= 0
			i
			die die die
		end
	end


	if size(X,1) ~= 50000
		die die die
	end
	
	if size(y,1) ~= 50000
		die die die
	end


	fprintf(1,'training svm\n')

	W = train_svm(X,y,C);


	fprintf('Loading test data...\n');
	f1=load([CIFAR_DIR '/test_batch.mat']);
	test_y = double(f1.labels) + 1;
	clear f1;

	test_X = load(test_features_path);
	try
		test_X = X.X;
		fprintf(1,'augmenting features\n')
		test_X = [ test_X, ones(size(test_X,1),1) ];
	catch
		test_X = test_X.testXCs;
	end


	[vals, pred_y ]  = max(test_X*W, [], 2);

	test_acc = mean(pred_y == test_y)
end
