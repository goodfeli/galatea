function [B S stat] = sparse_coding(X_total, num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save, pars, Binit, resample_size)
% Fast sparse coding algorithms
%
%    minimize_B,S   0.5*||X - B*S||^2 + beta*sum(abs(S(:)))
%    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
% 
% The detail of the algorithm is described in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007
%
% Written by Honglak Lee <hllee@cs.stanford.edu>
% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng
%
% options:
% X_total: training set 
% num_bases: number of bases
% beta: sparsity penalty parameter
% sparsity_func: sparsity penalty function ('L1', or 'epsL1')
% epsilon: epsilon for epsilon-L1 sparsity
% num_iters: number of iteration
% batch_size: small-batch size
% fname_save: filename to save
% pars: additional parameters to specify (see the code)
% Binit: initial B matrix
% resample_size: (optional) resample size 

if exist('resample_size', 'var') && resample_size
    assert(size(X_total,2) > resample_size);
    X = X_total(:, randsample(size(X_total,2), resample_size));
else
    X = X_total;
end

pars.patch_size = size(X,1);
pars.num_patches = size(X,2);
pars.num_bases = num_bases;
pars.num_trials = num_iters;

if exist('batch_size', 'var') && ~isempty(batch_size)
    pars.batch_size = batch_size; 
else
    pars.batch_size = size(X,2)/10;
end

pars.sparsity_func = sparsity_func;
pars.beta = beta;
pars.epsilon = epsilon;

pars.noise_var = 1;
pars.sigma = 1;
pars.VAR_basis = 1;

if ~isfield(pars,'display_images')	pars.display_images = false; end; %true;	
if ~isfield(pars,'display_every')	pars.display_every = 1;	end;
if ~isfield(pars,'save_every')	pars.save_every = 1;	end;
if ~isfield(pars,'save_basis_timestamps')	pars.save_basis_timestamps = true;	end;

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('../results/sc_b%d_%s', num_bases, datestr(now, 30));	
end;

% Sparsity parameters
if ~isfield(pars,'tol');                 pars.tol = 0.005; end;

% L1 sparsity function
if strcmp(pars.sparsity_func,'epsL1')
    pars.epsilon = epsilon;
    pars.reuse_coeff = false;
else
	pars.epsilon = [];
    pars.reuse_coeff = true;
end;

pars

% set path
addpath sc2

% initialize basis
if ~exist('Binit') || isempty(Binit)
    B = rand(pars.patch_size,pars.num_bases)-0.5;
	B = B - repmat(mean(B,1), size(B,1),1);
    B = B*diag(1./sqrt(sum(B.*B)));
else
    disp('Using Binit...');
    B = Binit;
end;
[L M]=size(B);
winsize=sqrt(L);

% initialize display
if pars.display_images
    figure(1), display_network_nonsquare2(B);
    % figure(1), colormap(gray);
end

S_all = zeros(M, pars.num_patches);

% initialize t only if it does not exist
if ~exist('t')	
	t=0;
	
	% statistics variable
	stat= [];
	stat.fobj_avg = [];
	stat.fresidue_avg = [];
	stat.fsparsity_avg = [];
	stat.var_tot = [];
	stat.svar_tot = [];
	stat.elapsed_time=0;
else
	% make sure that everything is continuous
	t= length(stat.fobj_avg)-1; 
end


% optimization loop
while t < pars.num_trials
    t=t+1;
    start_time= cputime;
    
    stat.fobj_total=0;
    stat.fresidue_total=0;
    stat.fsparsity_total=0;
    stat.var_tot=0;
    stat.svar_tot=0;

    if exist('resample_size', 'var') && resample_size
        fprintf('resample X (%d out of %d)...\n', resample_size, size(X_total,2));
        X = X_total(:, randsample(size(X_total,2), resample_size));
    end
    
    % Take a random permutation of the samples
    indperm = randperm(size(X,2));
    
    for batch=1:(size(X,2)/pars.batch_size),
        % Show progress in epoch
        if 1, fprintf('.'); end
        if (mod(batch,20)==0) fprintf('\n'); end
        
        % This is data to use for this step
        batch_idx = indperm((1:pars.batch_size)+pars.batch_size*(batch-1));
        Xb = X(:,batch_idx);
        
        % learn coefficients (conjugate gradient)
        if t==1 || ~pars.reuse_coeff
            if strcmp(pars.sparsity_func, 'L1') || strcmp(pars.sparsity_func, 'LARS') || strcmp(pars.sparsity_func, 'FS')
                S= l1ls_featuresign(B, Xb, pars.beta/pars.sigma*pars.noise_var);
            else
                S= cgf_fitS_sc2(B, Xb, pars.sparsity_func, pars.noise_var, pars.beta, pars.epsilon, pars.sigma, pars.tol, false, false, false);
            end
            S(find(isnan(S)))=0;
            S_all(:,batch_idx)= S;
        else
            if strcmp(pars.sparsity_func, 'L1') || strcmp(pars.sparsity_func, 'LARS') || strcmp(pars.sparsity_func, 'FS')
                tic
                S= l1ls_featuresign(B, Xb, pars.beta/pars.sigma*pars.noise_var, S_all(:,batch_idx));
                FS_time = toc
            else
                S= cgf_fitS_sc2(B, Xb, pars.sparsity_func, pars.noise_var, pars.beta, pars.epsilon, pars.sigma, pars.tol, false, false, false, S_all(:,batch_idx));
            end
            S(find(isnan(S)))=0;
            S_all(:,batch_idx)= S;
        end
        
        if strcmp(pars.sparsity_func, 'L1') || strcmp(pars.sparsity_func, 'LARS') || strcmp(pars.sparsity_func, 'FS')
            sparsity_S = sum(S(:)~=0)/length(S(:));
            fprintf('sparsity_S = %g\n', sparsity_S);
        end
        
        % get objective
        [fobj, fresidue, fsparsity] = getObjective2(B, S, Xb, pars.sparsity_func, pars.noise_var, pars.beta, pars.sigma, pars.epsilon);
        
        stat.fobj_total      = stat.fobj_total + fobj;
        stat.fresidue_total  = stat.fresidue_total + fresidue;
        stat.fsparsity_total = stat.fsparsity_total + fsparsity;
        stat.var_tot         = stat.var_tot + sum(sum(S.^2,1))/size(S,1);
        
        % update basis
        B = l2ls_learn_basis_dual(Xb, S, pars.VAR_basis);
    end
    
    % get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.num_patches;
    stat.fresidue_avg(t)  = stat.fresidue_total / pars.num_patches;
    stat.fsparsity_avg(t) = stat.fsparsity_total / pars.num_patches;
    stat.var_avg(t)       = stat.var_tot / pars.num_patches;
    stat.svar_avg(t)      = stat.svar_tot / pars.num_patches;
    stat.elapsed_time(t)  = cputime - start_time;
    
    % display
    if (pars.display_images && mod(t, pars.display_every)==0) || mod(t,pars.save_every)==0 || t==pars.num_trials
        display_figures(pars, stat, B, S, t);
    end
    
    fprintf(['epoch= %d, fobj= %f, fresidue= %f, fsparsity= %f, took %0.2f ' ...
             'seconds\n'], t, stat.fobj_avg(t), stat.fresidue_avg(t), ...
            stat.fsparsity_avg(t), stat.elapsed_time(t));
    
    % save results
    if mod(t,pars.save_every)==0 || t==pars.num_trials
        fprintf('saving results ...\n');
        experiment = [];
        experiment.matfname = sprintf('%s.mat', pars.filename);
        
        if pars.display_images
            save_figures(pars, t);
        end
        
        save(experiment.matfname, 't', 'pars', 'B', 'stat');
        fprintf('saved as %s\n', experiment.matfname);
    end
end

return

%% 

function retval = assert(expr)
retval = true;
if ~expr 
    error('Assertion failed');
    retval = false;
end
return
