clc;
close all;
clear all;


% obj = readObj('coarse_bolt_v3_size_reduced.obj');
% trimesh(obj.f.v, obj.v(:, 1), obj.v(:, 2), obj.v(:, 3))
% obj_f_v = obj.f.v;
% obj_v = obj.v;
% obj_vn = obj.vn;

load('obj_data.mat')
obj_f_v = new_face;
obj_v = new_pnt;
obj_vn = new_normal;

%% normal generation
obj_f_vn = zeros(size(obj_f_v,1),3);
for j = 1:size(obj_f_vn,1)
    p1 = obj_v(obj_f_v(j,1),:) - obj_v(obj_f_v(j,2),:);
    p2 = obj_v(obj_f_v(j,1),:) - obj_v(obj_f_v(j,3),:);
    n_temp = cross(p1,p2);
    n_temp = n_temp/norm(n_temp);
    
    z_hat = [0;0;1];
    theta =acos(dot(n_temp,z_hat));
    if theta > pi - 0.5*pi/180 || theta < 0.5*pi/180
        p3 = (obj_v(obj_f_v(j,1),:) + obj_v(obj_f_v(j,2),:) + obj_v(obj_f_v(j,3),:) )/3 - [0 0 0.03];
        if dot(n_temp,p3) > 0
            obj_f_vn(j,:) = n_temp;
        else
            obj_f_vn(j,:) = -n_temp;
        end        
    else
        p3 = (obj_v(obj_f_v(j,1),:) + obj_v(obj_f_v(j,2),:) + obj_v(obj_f_v(j,3),:) )/3;
        p3(3) = 0;
        if dot(n_temp,p3) > 0
            obj_f_vn(j,:) = n_temp;
        else
            obj_f_vn(j,:) = -n_temp;
        end        
    end
    
end

index = [];
for i = 1:size(obj_f_v,1)
    position = [0, 0, 0];
    for j = 1:3
        position = position + obj_v(obj_f_v(i, j), :);
    end
    position = position / 3;
    if position(3) < 0.039 && position(3) > 0.021 && (obj_f_vn(i,3) > 0.999 || obj_f_vn(i,3) < -0.999)
        index = [index;i];
    end
end
for i = 1:numel(index)
    position = [0, 0, 0];
    for j = 1:3
        position = position + obj_v(obj_f_v(index(i), j), :);
    end
    position = position / 3;
    obj_f_vn(index(i),:) = [position(1);position(2);0];
    obj_f_vn(index(i),:) = obj_f_vn(index(i),:)/norm(obj_f_vn(index(i),:));
end


%% parameter
noise_size = 0.001;
noise_size2 = 0.005;
eps = 1e-10;
data_num = 1000000;
input = zeros(data_num, 3);
output = zeros(data_num, 1);

%% for check
clc;
plus_pnts = zeros(data_num,3);
minus_pnts = zeros(data_num,3);
near_pnts = zeros(data_num,3);

plus_penet = zeros(data_num,1);
minus_penet = zeros(data_num,1);
near_penet = zeros(data_num,1);

% %% point generation
z_range = [0.02, 0.042];
r_range = [0.018, 0.026];
z_range2 = [0.038, 0.042];
r_range2 = [0, 0.018];
tic
parfor i_tmp = 1:data_num
    pnt_num = size(obj_v, 1);
    face_num = size(obj_f_v, 1);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55    
    fraction = (z_range2(2)-z_range2(1))/(z_range(2)-z_range(1)) * ...
        (r_range2(2)-r_range2(1))/(r_range(2)-r_range(1));
    if i_tmp < floor(data_num*fraction)
        u = 0.018* (rand(1) + rand(1));
        if u > 0.018
            r = 2 * 0.018 - u;
        else
            r = u;
        end
        theta = 2 * pi * rand(1);
        z = z_range(1) + (z_range(2) - z_range(1)) * rand(1);
        pnt = [r * cos(theta), r * sin(theta), z];
    else
        theta = rand(1) * 2 * pi;
        radius = r_range(1) + (r_range(2) - r_range(1)) * rand(1);

        pnt = [radius * cos(theta) ...
             radius * sin(theta), ...
            z_range(1) + (z_range(2) - z_range(1)) * rand(1)];
    end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55

    input(i_tmp, :) = pnt;
    
    min_dist = 10000;
    min_idx = -1;
    min_pnt = [100;100;100];
    min_geo = 2;
    min_s = 0;
    min_t = 0;
    penet = 10000;
    min_idx_same = 1;
    for j = 1:face_num
        % pnts in each row (pnt1 = tri(1, :))
        tri = zeros(3, 3);
        for k = 1:3
            tri(k, :) = obj_v(obj_f_v(j, k), :);
        end
        [closest_pnt, dist, geo, s, t] = distBtwPntTri(tri, pnt);
        if (dist < min_dist)
            min_dist = dist;
            min_idx = j;
            min_pnt = closest_pnt;
            min_geo = geo;
            min_s = s;
            min_t = t;
        end
    end
    if min_geo == 0
        % point
        % find index of point        
        if (min_s==0 && min_t ==0)
            idx = obj_f_v(min_idx, 1);
        elseif (min_s==0 && min_t == 1)
            idx = obj_f_v(min_idx, 3);
        else
            idx = obj_f_v(min_idx, 2);
        end
        a1 = find(obj_f_v(:,1) == idx);
        a2 = find(obj_f_v(:,2) == idx);
        a3 = find(obj_f_v(:,3) == idx);
        min_idx_same = union(a1,union(a2,a3));
    elseif min_geo == 1
        % line
        % find index of line        
        if (min_s + min_t == 1)
            idx1 = obj_f_v(min_idx, 2);
            idx2 = obj_f_v(min_idx, 3);
        elseif (min_s == 0)
            idx1 = obj_f_v(min_idx, 1);
            idx2 = obj_f_v(min_idx, 3);
        else
            idx1 = obj_f_v(min_idx, 1);
            idx2 = obj_f_v(min_idx, 2);
        end
        a1 = find(obj_f_v(:,1) == idx1);
        a2 = find(obj_f_v(:,2) == idx1);
        a3 = find(obj_f_v(:,3) == idx1);
        min_idx_same_temp = union(a1,union(a2,a3));
        obj_f_v_temp = obj_f_v(min_idx_same_temp,:);
        a1 = find(obj_f_v_temp(:,1) == idx2);
        a2 = find(obj_f_v_temp(:,2) == idx2);
        a3 = find(obj_f_v_temp(:,3) == idx2);
        min_idx_same_temp2 = union(a1,union(a2,a3));
        min_idx_same = min_idx_same_temp(min_idx_same_temp2);
    else
        min_idx_same = min_idx;        
    end
    
    if min_geo == 2 || min_geo == 0
        if numel(min_idx_same) == 1
            normal = obj_f_vn(min_idx_same,:);
        else
            normal = sum(obj_f_vn(min_idx_same,:));
        end
        normal = normal/norm(normal);
        penet = dot(normal, pnt - min_pnt); % plus: outside, minus: inside
    end
    
    if min_geo == 0
        normal1 = obj_f_vn(min_idx_same(1),:);
        penet_temp = dot(normal1(1:2),pnt(1:2)-min_pnt(1:2));
        if pnt(3) <= 0.039 && pnt(3) >= 0.021
            if sign(penet_temp) ~= sign(penet)
                penet = penet_temp
            end
        end
    end
    
    if min_geo == 1
        I_final = 1;
        for l = 1:numel(min_idx_same)-1
            position1 = [0, 0, 0];
            position2 = [0, 0, 0];
            for ll = 1:3
                position1 = position1 + obj_v(obj_f_v(min_idx_same(l), ll), :);
                position2 = position2 + obj_v(obj_f_v(min_idx_same(l+1), ll), :);
            end
            normal1 = obj_f_vn(min_idx_same(l),:);
            normal2 = obj_f_vn(min_idx_same(l+1),:);
            position1 = position1 / 3;
            position2 = position2 / 3;
            check1 = sign((pnt - position1)*normal1.') ~= sign((position2 - position1)*normal1.');
            check2 = sign((pnt - position2)*normal2.') ~= sign((position1 - position2)*normal2.');
            if check1 == 1 && check2 == 0
                I_final = l;
            elseif check1 == 0 && check2 == 1
                I_final = l+1;
            elseif (check1 == 0 && check2 == 0) || (check1 == 1 && check2 == 1)
                normal1 = obj_f_vn(min_idx_same(l),:);
                normal2 = obj_f_vn(min_idx_same(l+1),:);
                penet1 = dot(normal1, pnt - min_pnt); % plus: outside, minus: inside
                penet2 = dot(normal2, pnt - min_pnt); % plus: outside, minus: inside
                if sign(penet1) ~= sign(penet2)
                    penet_temp = dot(normal1(1:2),pnt(1:2)-min_pnt(1:2));
                    if sign(penet1) == sign(penet_temp)
                        I_final = l;
                    else
                        I_final = l+1;
                    end
                else
                    if abs(penet1) > abs(penet2)
                        I_final = l;
                    else
                        I_final = l+1;
                    end
                end
            end
        end
        
        normal = obj_f_vn(min_idx_same(I_final),:);
        penet = dot(normal, pnt - min_pnt); % plus: outside, minus: inside
    end
    output(i_tmp) = penet;

    if (penet > 0)
        plus_pnts(i_tmp,:) = pnt;
        plus_penet(i_tmp,:) = penet;
    else
        minus_pnts(i_tmp,:) = pnt;
        minus_penet(i_tmp,:) = penet;
    end

end
t_calc = toc
% s_near_pnts = sum(near_pnts.');
% i_near_pnts = find(s_near_pnts == 0);
% near_pnts(i_near_pnts,:) = [];

s_plus_pnts = sum(plus_pnts.');
i_plus_pnts = find(s_plus_pnts == 0);
plus_pnts(i_plus_pnts,:) = [];
plus_penet(i_plus_pnts,:) = [];

s_minus_pnts = sum(minus_pnts.');
i_minus_pnts = find(s_minus_pnts == 0);
minus_pnts(i_minus_pnts,:) = [];
minus_penet(i_minus_pnts,:) = [];

num_plus = size(plus_pnts,1)
num_minus = size(minus_pnts,1)
% num_near = size(near_pnts,1)


%% data check
close all
trimesh(obj_f_v, obj_v(:, 1), obj_v(:, 2), obj_v(:, 3))
hold on;
% plot3(plus_pnts(:, 1), plus_pnts(:, 2), plus_pnts(:, 3),  '.', 'Color', 'g');
plot3(minus_pnts(:, 1), minus_pnts(:, 2), minus_pnts(:, 3), '.', 'Color', 'r');
% plot3(near_pnts(:, 1), near_pnts(:, 2), near_pnts(:, 3), '.', 'Color', 'r');
xlabel('x')
ylabel('y')

% tmp = closest_pnt_normals;
% tmp2 = closest_pnts;
% tmp3 = same_dist_pnts;
% quiver3(tmp2(:, 1), tmp2(:, 2), tmp2(:, 3), tmp(:, 1), tmp(:, 2), tmp(:, 3),'k')

hold off;

%%
N_data = 1000000;
input_data = input(1:N_data,:).';
output_data = output(1:N_data,:).';
% M_input = max(input);
% m_input = min(input);
% M_output = max(output);
% m_output = min(output);
% for i = 1:N_data
%     input_data(:,i) = ((input(i,:) - m_input)./(M_input - m_input)).';
%     output_data(:,i) = ((output(i,:) - m_output)./(M_output - m_output)).';
% end

%% training
net = feedforwardnet([50 50 50 50]);
net.trainFcn = 'trainscg';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;
net.trainParam.epochs = 10000;
net.trainParam.min_grad = 1e-15;
net.trainParam.max_fail = 500;
net.inputs{1}.processFcns = { 'removeconstantrows', 'mapminmax' };
net.outputs{2}.processFcns =  { 'removeconstantrows', 'mapminmax' };
net.performParam.normalization = 'standard';
net.performParam.regularization = 0.1;
net.performFcn = 'mse';

% [net, tr] = train(net,input.',output.','useParallel','yes','showResources','yes');
[net, tr] = train(net,input_data,output_data,'useGPU','yes','showResources','yes');

%% test
tic
output_net = net(input_data);
toc
[~,index] = sort(output_data);

% output_net_sort = output_net(index).*(M_output - m_output) + m_output;
% output_true_sort = output_data(index).*(M_output - m_output) + m_output;
output_net_sort = output_net(index);
output_true_sort = output_data(index);

plot(output_net_sort,'.','color','r')
hold on
plot(output_true_sort,'.','color','k')
