function display_figures(pars, stat, A, S, t)
ws = sqrt(size(A,1));
figure(1), display_network_nonsquare2(A); % display_network(A,[],1); 
title(sprintf('%d iteration (%g min)', t, sum(stat.elapsed_time)/60));

figure(3); plot([1:t], stat.fobj_avg, [1:t], stat.fresidue_avg, [1:t], stat.fsparsity_avg); 
legend('fobj avg', 'fresidue avg', 'fsparsity avg'); 
title(sprintf('fobj=%f -> fobj=%f, %d iterations', stat.fobj_avg(1), stat.fobj_avg(end), length(stat.fobj_avg)));

if ~isempty(S)
    figure(5); plot(sort(abs(S(:))));
    if strcmp(pars.sparsity_func,'huberL1')
        figure(6); hist(pars.beta.*huber_func(S(:)/pars.sigma, pars.epsilon),30);
    elseif strcmp(pars.sparsity_func,'epsL1')
        figure(6); hist(pars.beta.*sqrt(pars.epsilon+(S(:)/pars.sigma).^2),30);
    else
        figure(6); hist(pars.beta.*log(1+(S(:)/pars.sigma).^2),30);
    end
end
