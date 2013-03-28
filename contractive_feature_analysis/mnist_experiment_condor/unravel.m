function [A,b,c] = unravel(W,mu)

if size(mu,2) ~=1
	mu = mu';
end

[h nn] = size(W);
%nn = n+n*(n+1)/2
%so 
n = (sqrt(8*nn+9)-3)/2;


c = - W * mu;
b = W(:,1:n);

A = {};

for k=1:h
	idx = n+1;
	
	A{k} = zeros(n,n);
	
	for i=1:n
	%fprintf('expansions involving %d\n',i);
		for j=i:n
			%[idx nn]
			A{k}(i,j) = A{k}(i,j) + 0.5 *W(k,idx);
			A{k}(j,i) = A{k}(j,i) + 0.5 * W(k,idx);
			idx = idx + 1;
		end
	end
end

if idx - 1 ~= nn
	error('oh no!')
end

end
