function show_optimal_quadratic(W, whitening_basis, mu2, pca_basis,mu, nrm, tol, varargin),
% SHOW_OPTIMAL shows the optimal excitatory stimuli of an SFA object.
%
% SHOW_OPTIMAL(HDL, NRM, TOL) Show the optimal excitatory stimuli of SFA
%   object HDL, under a fixed norm constraint. The fixed norm is specified
%   by NRM. TOL gives the maximal norm difference tollerated by the
%   maximization algorithm.
%
%   Optional arguments can be specified by appending
%   'ArgumentName',ArgumentValue pairs to the argument list
%   (e.g. SHOW_OPTIMAL(HDL,NRM,TOL,'start',5,'show_xm',1) ).
%
%   Possible optional arguments:
%   'show_xm' (default:0) if set to 1 shows the optimal inhibitory stimuli
%
%   'h' (default:16) height of the input patch
%   'w' (default:16) width of the input patch
%
%   'sh' (default:7)
%   'sw' (default:7) the optimal stimuli are displayed on a SH times SW
%                    grid
%
%   'start' (default:1) the first unit to consider
  

  %%%% default values

addpath ~/berkes/sfa_tk/lcov
addpath ~/berkes/sfa_tk/sfa
addpath ~/berkes/slowness_model
Aw = W * whitening_basis;
[Ap, bp, cp] = unravel(Aw,mu2);

if size(mu,2) ~= 1
	mu = mu';
end

  % set to 1 if you want to see the optimal inhibitory stimuli
  ctxt.show_xm = 0;

  % input patch height and width
  ctxt.h = 16; ctxt.w = 16;

  % number of optimal stimuli to display (vertically and horizontally)
  ctxt.sh = 7; ctxt.sw = 7;
  % first unit to consider
  ctxt.start = 1;
  % default window position
  ctxt.Position = [360 372 597 562]; % NxN

  % overwrite with user-defined list of settings
  for k = 1:2:length(varargin);
    % error check: the optional arguments must be defined as name-value pairs
    if ~ischar(varargin{k}),
      error 'Setting names must be strings';
    end
    % set variable value
    ctxt=setfield(ctxt,varargin{k},varargin{k+1});
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % range of units to consider
  range = ctxt.start:min([size(W,1) (ctxt.start+ctxt.sh*ctxt.sw-1)]);

  % copy some useful quantities
  global SFA_STRUCTS
  %avg = SFA_STRUCTS{hdl}.avg0;
  %DW = SFA_STRUCTS{hdl}.DW0;
  %D = SFA_STRUCTS{hdl}.D0;
  %PCinv = DW*diag(D.^(-0.5));
  PCinv = pca_basis';

  % set the figure up
  clf; colormap(gray);
  set(gcf, 'Position', ctxt.Position);
  
  % subplot counter
  k=1;

  currow = 1;
  curcol = 1;

  % loop over all subunits
  for i=range,
    % get the quadratic form corresponding to the current subunit in the
    % principal components space (much faster then in the input space).
    % ! technical detail: this is equivalent to finding the optimal
    % stimuli in the input _meanfree_ space.
    % This is similar to physiological experiments where the optimal stimuli
    % are defined as intensity changes with respect to a constant mean
    % luminance (and thus have negative and positive intensity values).

    fprintf('i=%d\n',i)
    
    H = Ap{i}*2;
    f = bp(i,:)';
    c =  cp(i);
   

    fprintf('running maximization\n')

    % get the maximum and the minimum of the quadratic form
    if i ~= 110 && i ~= 210 && i ~= 265 && i~= 269
        xp = maximize_qform(H,f, [], nrm, tol);
    end

    fprintf('finished first\n');


    if i == 3 || i == 187 || i == 243 
	    xm = -xp
	    'hack'
    else
    	xm = maximize_qform(-H,-f, [], nrm, tol);
    end
    vm = 0.5*xm'*H*xm + f'*xm + c;

    if i == 110 || i == 210 || i == 265 || i == 269
	    xp = -xm;
    end
    vp = 0.5*xp'*H*xp + f'*xp + c;

    fprintf('done running maximization\n')

    % xp has the largest value (or xm if show_xm==1)
    if (abs(vm)>vp & ~ctxt.show_xm) | (abs(vm)<vp & ctxt.show_xm),
      tmp = xp; xp = xm; xm = tmp;
    end;

    % project xp back to the input space
    tmp = PCinv*xp;
    xp = tmp+mu;
    xm = PCinv*xm+mu;

    xp = xp - min(xp);
    xp = xp / max(xp);
    xp = reshape(xp, ctxt.h, ctxt.w);

    if curcol == 1
	rowimage = xp;
    else
	rowimage = [rowimage zeros(ctxt.h,3) xp];
    end

    if curcol == ctxt.sw
	curcol = 0;

	if currow == 1
		image = rowimage;
	else
		image = [image; zeros(3, size(image,2)); rowimage ];
	end 

	currow = currow + 1;
    else
	curcol = curcol + 1;
    end

    % plot
    %subplot(ctxt.sh,ctxt.sw,k);
    %imagesc(reshape(xp, ctxt.h, ctxt.w));
    %axis off; axis image; drawnow,

    % increase the subplot counter
    %k = k+1
  end

  save('fuck_you.mat','image')

  imshow(image)
  
  
function x = maximize_qform(H,f, x0, nrm, tol),
% maximize the quadratic form 1/2*x'*H*x + f'*x + c
  
  % center the quadratic form around x0
  if ~isempty(x0),
    f = H*x0+f;
    %c = 0.5*x0'*H*x0 + f'*x0 + c;
  end
  
  % input dimension
  dim = size(H,1);
  % norm of f
  nrm_f = norm(f);

  % get eigenvalues and eigenvectors  
  [V,D] = eig(H);

  mu = diag(D)';
  % coefficients of the eigenvectors decomposition of f
  alpha = V'*f;
  % v_i = alpha_i * v_i
  V=V.*repmat(alpha',dim,1);

  % left bound for lambda
  % added 'real' to avoid numerical problems if you maximize in input space
  ll = max(real(mu));
  % right bound for lambda
  lr = norm(f)/nrm + ll;
  
  % search by bisection until norm(x)^2 = nrm^2
  nrm_2 = nrm^2;
  norm_x_2 = 0;
  while abs(norm_x_2-nrm_2)>tol,
    % bisection of the lambda-interval
    lambda=(lr-ll)/2+ll;
    % eigenvalues of (lambda*Id - H)^-1
    beta = (lambda-mu).^(-1);
	  
    % solution to the second lagragian equation
    norm_x_2 = sum(alpha.^2.*beta'.^2);

    %[ll,lr]
    if norm_x_2>nrm_2, ll=lambda;
    else lr=lambda; end
    %[ll, lr, norm_x_2]
    %pause(1)
  end
  
  x = sum(V.*repmat(beta,dim,1),2);
  
  if ~isempty(x0),
    x=x+x0;
  end
