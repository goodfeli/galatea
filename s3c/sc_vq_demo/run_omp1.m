function dictionary = run_omp1(X, k, iterations)

  % initialize dictionary
  dictionary = randn(k, size(X,2));
  dictionary = bsxfun(@rdivide, dictionary, sqrt(sum(dictionary.^2,2)+1e-20));
  
  for itr=1:iterations
    fprintf(1,'Running GSVQ:  iteration=%d... \n', itr);

    % do assignment + accumulation
    [summation,counts] = gsvq_step(X, dictionary);

    % reinit empty clusters
    I=find(sum(summation.^2,2) < 0.001);
    summation(I,:) = randn(length(I), size(X,2));

    % normalize
    dictionary = bsxfun(@rdivide, summation, sqrt(sum(summation.^2,2)+1e-20));
  end

  
function [summation, counts] = gsvq_step(X, dictionary)

  summation = zeros(size(dictionary));
  counts = zeros(size(dictionary,1),1);

  k = size(dictionary,1);

  BATCH_SIZE=2000;

  tic;
  for i=1:BATCH_SIZE:size(X,1)
    lastInd=min(i+BATCH_SIZE-1, size(X,1));
    m = lastInd - i + 1;

    dots = dictionary*X(i:lastInd,:)';
    [val,labels] = max(abs(dots)); % get labels

    E = sparse(labels,1:m,1,k,m,m); % labels as indicator matrix
    counts = counts + sum(E,2);  % sum up counts

    dots = dots .* E; % dots is now sparse
    summation = summation + dots * X(i:lastInd,:); % take sum, weighted by dot product
  end
