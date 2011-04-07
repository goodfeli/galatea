function [fobj, fresidue, fsparsity] = getObjective2(A, S, X, sparsity, noise_var, beta, sigma, epsilon)


if ~strcmp(sparsity, 'log') && ~strcmp(sparsity, 'huberL1') && ~strcmp(sparsity,'epsL1') && ...
        ~strcmp(sparsity,'FS') && ~strcmp(sparsity, 'L1') && ~strcmp(sparsity,'LARS') && ...
        ~strcmp(sparsity, 'trueL1') && ~strcmp(sparsity, 'logpos')
	error('sparsity function is not properly specified!\n');
end

if strcmp(sparsity, 'huberL1') || strcmp(sparsity, 'epsL1')
	if ~exist('epsilon','var') || isempty(epsilon) || epsilon==0
		error('epsilon was not set properly!\n')
	end
end


E = A*S - X;
lambda=1/noise_var;
fresidue  = 0.5*lambda*sum(sum(E.^2));

if strcmp(sparsity, 'log')
	fsparsity = beta*sum(sum(log(1+(S/sigma).^2)));
elseif strcmp(sparsity, 'huberL1') 
	fsparsity = beta*sum(sum(huber_func(S/sigma, epsilon)));
elseif strcmp(sparsity, 'epsL1')
	fsparsity = beta*sum(sum(sqrt(epsilon+(S/sigma).^2)));
elseif strcmp(sparsity, 'L1') | strcmp(sparsity, 'LARS') | strcmp(sparsity, 'trueL1') | strcmp(sparsity, 'FS')		  
    fsparsity = beta*sum(sum(abs(S/sigma)));
elseif strcmp(sparsity, 'logpos')
	fsparsity = beta*sum(sum(log(1+(S/sigma))));
end

fobj = fresidue + fsparsity;
