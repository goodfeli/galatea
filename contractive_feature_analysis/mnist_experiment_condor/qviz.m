clear
load matlab_dump

%correct for difference in formatting between Ian's CFA code and Berkes' SFA code
pca_basis = pca_basis';
W = W';
whitening_basis = whitening_basis';

W = W(1:289,:);

n = size(pca_basis,2);
w = sqrt(n);
h = size(W,1);
sh = floor(sqrt(h));
sw = sh;
while sw * sh < h
sw = sw + 1;
end

'patch width'
w
'# hu'
h

show_optimal_quadratic(W, whitening_basis, mu2, pca_basis,mu, 1.0, 1e-3, 'w',w,'h',w,'sh', sh,'sw',sw)
