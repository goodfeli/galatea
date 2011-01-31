function []=test_case(name, data, test_data)

heb = hebbian;

fprintf('Test case "%s"\n', name);

fprintf('Non-kernelized version\n');
%fprintf('Before training\n');
%fprintf('b0:\n');
%heb.b0
%fprintf('W:\n');
%heb.W

[res, heb] = train(heb, data);

fprintf('After training\n');
fprintf('b0:\n');
heb.b0
fprintf('W:\n');
heb.W

preds = test(heb, test_data);

fprintf('preds:\n');
preds.X


fprintf('Kernelized version\n');

heb = hebbian;

X = data.X;

% Avoid kernelizing twice if the data has already been 
% detected to be kernelized
if ~pd_check(data)
    data = kernelize(data);
end;

test_data.X = test_data.X * X';

%fprintf('Before training\n');
%fprintf('b0:\n');
%heb.b0
%fprintf('W:\n');
%heb.W

[res, heb] = train(heb, data);

fprintf('After training\n');
fprintf('b0:\n');
heb.b0
fprintf('W:\n');
heb.W

preds = test(heb, test_data);

fprintf('preds:\n');
preds.X

fprintf('End of "%s"\n', name);