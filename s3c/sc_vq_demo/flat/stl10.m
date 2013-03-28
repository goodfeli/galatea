
fprintf(1,'loading test features\n')
load feat_test.mat
test_X = [ X, ones(size(X,1),1) ];
fprintf(1,'loading test data\n')
load /data/lisa/data/stl10/stl10_matlab/test.mat
clear X
test_y = y;

fprintf(1,'loading stl10\n')
load /data/lisa/data/stl10/stl10_matlab/train.mat
clear X;
C = 100.;
fprintf(1,'loading features\n')
load feat.mat

fprintf(1,'augmenting features\n')
X = [X, ones(size(X,1),1)];

k = 1;
results = zeros(k,1);

for i = 1:k
	fprintf(1,'i = %d\n',i)
	fprintf(1,'training svm\n')
	fi = fold_indices{i};
	W = train_svm(X(fi,:),y(fi),C);

	'W'
	max(max(W))

	cls = test_X*W;

	mx = max(cls');
	mask = cls == repmat(mx',1,size(cls,2));


	pred_y = mask * [1:size(cls,2)]';


	test_acc = mean(pred_y == test_y)

	results(i) = test_acc 
end

'results'
mean(results)

