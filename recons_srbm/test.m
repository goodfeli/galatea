q = [0,0];
mu = [0,0];

beta = 1.;
W = [1,1];
alpha = 1.;
w = beta * ( W.^2);
gamma = alpha+w;
c=[0,0];
damp = 0.;
a=[0,0];

record = zeros(10,2);

for iter=1:10
	iter
	q
	mu

	record(iter,:) = q*mu;


	A_pre = w .* q .* mu - (beta * (w*w') * ( (q.*mu)'))'  + a;
	beta_W = beta * W;
	A = A_pre + beta_W;
	new_mu = A ./ gamma;
	arg = 0.5 * ((A.^2)./gamma-(a.^2)./alpha-log((alpha+w)./alpha)) + c;
	new_q = 1./(1.+exp(-arg));

	q = (1.-damp)*new_q + damp*q;
	mu = (1.-damp)*new_mu + damp*mu;

	iter = iter + 1;
end

plot(record(1,:))
hold on
plot(record(2,:))
