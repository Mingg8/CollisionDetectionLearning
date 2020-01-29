% close all;
clear

load('data.csv')
input = data(:, 1:3);
output = data(:, 4);
pred_o = data(:, 5);
pred_g = data(:, 6:8);
pred_g2 = data(:, 9:11);

load('~/workspace/NutLearning/obj/data/obj_data.mat')

%%
trimesh(new_face, new_pnt(:,1), new_pnt(:,2), new_pnt(:,3))
hold on;
% plot3(input(:,1), input(:,2), input(:,3), '.', 'Color', 'b')
quiver3(input(:,1), input(:,2), input(:,3), ...
    pred_g(:, 1), pred_g(:, 2), pred_g(:, 3))

%%
pnt = [];
for i = 1:size(data, 1)
    if abs(output(i) - pred_o(i)) > 0.0005
        pnt = [pnt; input(i, :)];
    end
end

%%
% close all;
figure(3)
trimesh(new_face, new_pnt(:,1), new_pnt(:,2), new_pnt(:,3))
hold on;
plot3(pnt(:,1), pnt(:,2), pnt(:,3), '.', 'Color', 'b')