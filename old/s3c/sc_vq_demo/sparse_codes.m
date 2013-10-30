function out = sparse_codes(X, D, lambda)
  % set optimization parameters
  param.lambda=lambda;
  param.numThreads=-1; % use all cores
  param.mode=2;       % penalized formulation
  param.itermax = 1000;
  param.tol = 0.001;
  assert(all( abs(sum(D.^2,2) - 1) < 1e-10 ));
  out = full(mexCD(X',D', sparse(size(D,1),size(X,1)), param)');
