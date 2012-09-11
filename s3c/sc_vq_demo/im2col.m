function B = im2col(A,fuck_you)
	m = fuck_you(1);
	n = fuck_you(2);
	Ar = size(A,1);
	cr = Ar - m + 1;
	Ac = size(A,2);
	cc = Ac - n + 1;

	Br = m * n;
	Bc = cr * cc;

	B = zeros(Br,Bc);

	idx = 0;

	for c = 1:cc
		for r = 1:cr
			idx = idx + 1;
			B(:,idx) = reshape(A(r:r+m-1,c:c+n-1),[m*n,1]);
		end
	end
end
