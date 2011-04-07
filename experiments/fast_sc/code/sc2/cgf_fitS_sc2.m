function S = cgf_fitS_sc2(A,X, sparsity, noise_var, beta, epsilon, sigma, tol, disp_ocbsol, disp_patnum, disp_stats, Sinit)
% cgf_fitS -- fit internal vars S to the data X using fast congugate gradient
%   Usage
%     S = cgf_fitS(A,X,noise_var,beta,sigma,
%                  [tol, disp_ocbsol, disp_patnum, disp_stats])
%   Inputs
%      A             basis functions
%      X             data vectors
%      noise_var     variance of the noise (|x-As|^2)
%      beta          steepness term for prior
%      sigma         scaling term for prior
%      tol           solution tolerance (default 0.001)
%      disp_ocbsol   display info from the fitting process
%      disp_patnum   display the pattern number
%      disp_stats    display summary statistics for the fit
%   Outputs
%      S             the estimated coefficients

maxiter=100;

[L,M] = size(A);
N = size(X,2);

if ~exist('tol','var');		tol = 0.001;			end
if ~exist('disp_ocbsol','var');	disp_ocbsol = 0;		end
if ~exist('disp_patnum','var');	disp_patnum = 1;		end
if ~exist('disp_stats','var');	disp_stats = 1;			end
if ~exist('maxiter','var');		maxiter = 8;			end
if ~exist('reduction','var');	reduction = 8;			end

% XXX: we don't use initialization for "log" sparsity function because of local optima
if ~exist('Sinit','var') %|| strcmp(sparsity, 'log') || strcmp(sparsity, 'huberL1') || strcmp(sparsity, 'epsL1')
	Sinit=A'*X;
	normA2=sum(A.*A)';
	for i=1:N
        Sinit(:,i)=Sinit(:,i)./normA2;
	end
    initiated = 0;
else
    initiated = 1;
end

if ~strcmp(sparsity, 'log') && ~strcmp(sparsity, 'huberL1') && ~strcmp(sparsity, ...
                                                      'epsL1')
	error('sparsity function is not properly specified!\n');
end

lambda=1/noise_var;

if strcmp(sparsity, 'huberL1') || strcmp(sparsity, 'epsL1')
	if ~exist('epsilon','var') || isempty(epsilon) || epsilon==0
		error('epsilon was not set properly!\n')
	end
end

S = zeros(M,N);
tic
if ~initiated
    if strcmp(sparsity, 'log')
        [S niters nf ng] = cgf_sc2(A,X,Sinit,0,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum);
    elseif strcmp(sparsity, 'huberL1') 
        [S niters nf ng] = cgf_sc2(A,X,Sinit,1,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum, epsilon);
    elseif strcmp(sparsity, 'epsL1')
        [S niters nf ng] = cgf_sc2(A,X,Sinit,2,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum, epsilon);
    end
else
    if strcmp(sparsity, 'log')
        [S niters nf ng] = cgf_sc2(A,X,Sinit,0,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum);
    elseif strcmp(sparsity, 'huberL1') 
        [S niters nf ng] = cgf_sc2(A,X,Sinit,1,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum, epsilon);
    elseif strcmp(sparsity, 'epsL1')
        [S niters nf ng] = cgf_sc2(A,X,Sinit,2,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum, epsilon);
    end
%      for i=1:size(X,2)
%          [aa,bb] = sort(abs(Sinit(:,i)));
%          bb = flipud(bb);
%          active = bb(1:M/reduction);
%          if strcmp(sparsity, 'log')
%              [S2 niters nf ng] = cgf_sc2(A(:,active),X(:,i),Sinit(:,i),0,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum);
%          elseif strcmp(sparsity, 'huberL1') 
%              [S2 niters nf ng] = cgf_sc2(A(:,active),X(:,i),Sinit(:,i),1,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum, epsilon);
%          elseif strcmp(sparsity, 'epsL1')
%              [S2 niters nf ng] = cgf_sc2(A(:,active),X(:,i),Sinit(:,i),2,lambda,beta,sigma,tol,maxiter, disp_ocbsol,disp_patnum, epsilon);
%          end
%          S(active,i) = S2;
%      end
%      fprintf('%d',reduction);
end
t = toc;

if (disp_stats)
  fprintf(' aits=%6.2f af=%6.2f ag=%6.2f  at=%7.4f\n', ...
      niters/N, nf/N, ng/N, t/N);
end
