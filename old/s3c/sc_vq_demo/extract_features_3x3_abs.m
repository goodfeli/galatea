function XC = extract_features_abs(X, D, rfSize, CIFAR_DIM, M,P, encoder, encParam)
    numBases = size(D,1);
    
    % compute features for all training images
    XC = zeros(size(X,1), numBases*9);
    for i=1:size(X,1)
        if (mod(i,1000) == 0) fprintf('Extracting features: %d / %d\n', i, size(X,1)); end
        
        % extract overlapping sub-patches into rows of 'patches'
        patches = [ im2col(reshape(X(i,1:1024),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                    im2col(reshape(X(i,1025:2048),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                    im2col(reshape(X(i,2049:end),CIFAR_DIM(1:2)), [rfSize rfSize]) ]';

        % do preprocessing for each patch
        
        % normalize for contrast
        patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
        % whiten
        patches = bsxfun(@minus, patches, M) * P;
    
        % compute activation
        switch (encoder)
         case 'thresh'
          alpha=encParam;
          z = patches * D';
          patches = max(abs(z) - alpha, 0);
          clear z;
         case 'sc'
          lambda=encParam;
          z = sparse_codes(patches, D, lambda);
          patches = abs(z);
         otherwise
          error('Unknown encoder type.');
        end
        % patches is now the data matrix of activations for each patch
        
        % reshape to 2*numBases-channel image
        prows = CIFAR_DIM(1)-rfSize+1;
        pcols = CIFAR_DIM(2)-rfSize+1;
        patches = reshape(patches, prows, pcols, numBases);
        
        % pool over quadrants
        r1 = round(prows/3);
	r2 = 2 * r1;
        c1 = round(pcols/3);
	c2 = 2 * c1;
        q1 = sum(sum(patches(1:r1, 1:c1, :), 1),2);
        q2 = sum(sum(patches(r1+1:r2, 1:c1, :), 1),2);
	q3 = sum(sum(patches(r2+1:end, 1:c1, :), 1),2);
        q4 = sum(sum(patches(1:r1, c1:c2, :), 1),2);
        q5 = sum(sum(patches(r1+1:r2, c1:c2, :), 1),2);
	q6 = sum(sum(patches(r2+1:end, c1:c2, :), 1),2);
	q7 = sum(sum(patches(1:r1, c2:end, :), 1),2);
	q8 = sum(sum(patches(r1+1:r2, c2:end, :), 1),2);
	q9 = sum(sum(patches(r2+1:end, c2:end, :), 1),2);
        
        % concatenate into feature vector
        XC(i,:) = [q1(:);q2(:);q3(:);q4(:);q5(:);q6(:);q7(:);q8(:);q9(:)]';
    end

