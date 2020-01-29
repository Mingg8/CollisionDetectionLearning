function [y] = relu(x)
%RELU Summary of this function goes here
%   Detailed explanation goes here
y = zeros(numel(x), 1);
for i = 1:numel(x)
    if x(i) >= 0
        y(i) = x(i);
    else
        y(i) = 0;
    end 
end
end

