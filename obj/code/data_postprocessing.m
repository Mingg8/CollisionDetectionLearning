function [i_data, o_data] = data_postprocessing(filename)
%DATA_POSTPROCESSING Summary of this function goes here
%   Detailed explanation goes here
l = load(filename);
size(l.minus_pnts)
for i = 1:size(l.minus_pnts, 1)
    if norm(l.minus_pnts(i, :)) == 0
        break;
    end
end

for j = 1:size(l.plus_pnts, 1)
    if norm(l.plus_pnts(j, :)) == 0
        break;
    end
end

i_data = [l.plus_pnts; l.minus_pnts(1:i-1, :) ...
    ; l.plus_pnts(1:j-1, :)];
o_data = [l.plus_penet; l.minus_penet(1:i-1, :) ...
    ; l.plus_penet(1:j-1, :)];
end

