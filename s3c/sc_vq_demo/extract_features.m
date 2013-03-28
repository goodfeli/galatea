function XC = extract_features(X, D, rfSize, CIFAR_DIM, M,P, encoder, encParam)
    numBases = size(D,1);
    
    % compute features for all training images
    XC = zeros(size(X,1), numBases*2*4);
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
          patches = [ max(z - alpha, 0), -max(-z - alpha, 0) ];
          clear z;
         case 'sc'
          lambda=encParam;
          z = sparse_codes(patches, D, lambda);
          patches = [ max(z, 0), -min(z, 0) ];
         otherwise
          error('Unknown encoder type.');
        end
        % patches is now the data matrix of activations for each patch
        
        % reshape to 2*numBases-channel image
        prows = CIFAR_DIM(1)-rfSize+1;
        pcols = CIFAR_DIM(2)-rfSize+1;
        patches = reshape(patches, prows, pcols, numBases*2);
        
        % pool over quadrants
        halfr = round(prows/2);
        halfc = round(pcols/2);
        q1 = sum(sum(patches(1:halfr, 1:halfc, :), 1),2);
        q2 = sum(sum(patches(halfr+1:end, 1:halfc, :), 1),2);
        q3 = sum(sum(patches(1:halfr, halfc+1:end, :), 1),2);
        q4 = sum(sum(patches(halfr+1:end, halfc+1:end, :), 1),2);
        
        % concatenate into feature vector
        XC(i,:) = [q1(:);q2(:);q3(:);q4(:)]';
    end

