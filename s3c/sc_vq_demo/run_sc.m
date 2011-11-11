function dictionary = run_sc(X, k, iters, lambda)

  % initialize dictionary
  dictionary = randn(k, size(X, 2));
  dictionary = bsxfun(@rdivide, dictionary, sqrt(sum(dictionary.^2,2)+1e-20));
  for itr=1:iters
    fprintf(1,'Running sparse coding:  iteration=%d... \n', itr);
    S = sparse_codes(X, dictionary, lambda);
    dictionary = S \ X;
    dictionary = bsxfun(@rdivide, dictionary, sqrt(sum(dictionary.^2,2))+1e-20);
  end
  
