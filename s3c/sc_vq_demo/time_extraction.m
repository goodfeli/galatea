function XC = time_extraction(all_patches, D, lambda)
    numBases = size(D,1);
    
    z = sparse_codes(all_patches, D, lambda);

    % compute features for all training images
    %XC = zeros(size(all_patches,1), size(D,1));
    %for i=1:size(all_patches,1)
        
    %    patches = all_patches(i,:);

    %    z = sparse_codes(patches, D, lambda);

    %	XC(i,:) = z;
    %end

