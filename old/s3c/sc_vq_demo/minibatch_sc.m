function dictionary = minibatch_sc(X, k, iters, lambda, batch_size, new_coeff)

  % initialize dictionary
  dictionary = randn(k, size(X, 2));
  dictionary = bsxfun(@rdivide, dictionary, sqrt(sum(dictionary.^2,2)+1e-20));
  for itr=1:iters
    fprintf(1,'Running sparse coding:  iteration=%d... \n', itr);

    for batch_idx = 1:batch_size:(size(X,1))
	fprintf(1, '\ton index %d\n',batch_idx)
	batch = X(batch_idx:batch_idx+batch_size-1,:);

    	S = sparse_codes(batch, dictionary, lambda);

    	new_dict = S \ batch;
    	new_dict = bsxfun(@rdivide, dictionary, sqrt(sum(new_dict.^2,2))+1e-20);

    	dictionary = new_coeff * new_dict + (1.-new_coeff) * dictionary;

    	dictionary = bsxfun(@rdivide, dictionary, sqrt(sum(dictionary.^2,2))+1e-20);
  end
 end
end 
