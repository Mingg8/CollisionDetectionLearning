clear
input_final = [];
output_final = [];

%%
load('data1_3000.mat');
input_final = [input_final; minus_pnts; plus_pnts; near_pnts];
output_final = [output_final; minus_penet; plus_penet; near_penet];
size(output_final)

load('data2_3000.mat');
input_final = [input_final; minus_pnts; plus_pnts; near_pnts];
output_final = [output_final; minus_penet; plus_penet; near_penet];
size(output_final)

load('data3_3000.mat');
input_final = [input_final; minus_pnts; plus_pnts; near_pnts];
output_final = [output_final; minus_penet; plus_penet; near_penet];
size(output_final)

load('data4_3000.mat');
input_final = [input_final; minus_pnts; plus_pnts; near_pnts];
output_final = [output_final; minus_penet; plus_penet; near_penet];
size(output_final)

load('data5_3000.mat');
input_final = [input_final; minus_pnts; plus_pnts; near_pnts];
output_final = [output_final; minus_penet; plus_penet; near_penet];
size(output_final)

load('data6_5000.mat');
input_final = [input_final; minus_pnts; plus_pnts; near_pnts];
output_final = [output_final; minus_penet; plus_penet; near_penet];
size(output_final)

load('data_up_10000.mat');
input_final = [input_final; minus_pnts; plus_pnts; near_pnts];
output_final = [output_final; minus_penet; plus_penet; near_penet];
size(output_final)

%%

plus_pnt_final = [];
minus_pnt_final = [];

for i = 1:size(input_final, 1)
    if (output_final(i) < 0)
        minus_pnt_final = [minus_pnt_final; input_final(i, :)];
    end
end

%%
plot3(minus_pnt_final(:, 1), minus_pnt_final(:, 2), minus_pnt_final(:, 3), '.');
axis equal