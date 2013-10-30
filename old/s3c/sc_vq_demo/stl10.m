'loading test features'
load feat_test.mat
test_X = [ X, ones(size(X,1),1) ];
'loading test data'
load /data/lisa/data/stl10/stl10_matlab/test.mat
clear X
test_y = y;
addpath minFunc

'loading stl10'
load /data/lisa/data/stl10/stl10_matlab/train.mat
clear X;
C = 10000.;
'loading features'
load feat.mat

'augmenting features'
X = [X, ones(size(X,1),1)];

results = zeros(10,1);

for i = 1:10
	'training svm'
	fold_idxs = fold_indices{i};
	W = train_svm(double(X(fold_idxs,:)),y(fold_idxs),C);

	cls = test_X*W;
	mx = max(cls');
	mask = cls == repmat(mx',1,size(cls,2));
	pred_y = mask * [1:size(cls,2)]';

	test_acc = mean(pred_y == test_y)

	results(i) = test_acc 
end

mean(results)
