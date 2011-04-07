function save_figures(pars, t)

experiment=[];
experiment.basisfname = sprintf('%s_basis.png', pars.filename);
experiment.fobjfname = sprintf('%s_fobj.png', pars.filename);
experiment.histcoeffname = sprintf('%s_hist_coef.png', pars.filename);
experiment.histfspfname = sprintf('%s_hist_fsp.png', pars.filename);

figure(1), saveas(gcf, experiment.basisfname);
figure(3), saveas(gcf, experiment.fobjfname);
figure(5); saveas(gcf, experiment.histcoeffname);
figure(6); saveas(gcf, experiment.histfspfname);

if pars.save_basis_timestamps
	timestamp_basisfname = sprintf('%s_basis_%04dt.png', pars.filename, t);
	figure(1), saveas(gcf, timestamp_basisfname);
end
