%% weight load
for i = 0:7
    a = append('weight', int2str(i));
    a = append(a, '.csv');
    load(a);
end
input_coeff = load('input_coeff.csv');
output_coeff = load('output_coeff.csv');

%%
pnt = [0.0165, 0.0165, 0.0371];
i = input_coeff(1:3)' .* pnt + input_coeff(4:6)';
tic
o = tanh(weight7 + weight6' * relu(weight5 + weight4' * relu(weight3 + weight2' * ...
    relu(weight1 + weight0' * i'))));
toc
output = (o - output_coeff(5:8)) ./ output_coeff(1:4);
