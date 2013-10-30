function patches = extract_patches(X, rfSize, CIFAR_DIM, num_patches)

	patches = zeros( num_patches, rfSize * rfSize * 3);

	i = 1;
	j = 1;

	while 1
        	one_img_patches = [ im2col(reshape(X(i,1:1024),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                    im2col(reshape(X(i,1025:2048),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                    im2col(reshape(X(i,2049:end),CIFAR_DIM(1:2)), [rfSize rfSize]) ]';


		last = min(num_patches, j + size(one_img_patches,1) - 1);

		patches(j:last,:) = one_img_patches(1:last-j+1,:);

		if last == num_patches
			break
		end

		i = i + 1;
		j = last;
	end



