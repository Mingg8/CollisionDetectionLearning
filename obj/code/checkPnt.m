% pnt = [-0.0021342, -0.023638, 0.038864];
% pnt = [-0.019709, -0.013177, 0.038305];
% pnt = [-0.01847, 0.01496, 0.03433];
pnt = [-0.012694, 0.020162, 0.020093];
% pnt = [-0.0037253, 0.023566, 0.020433]];

%% calculate normal & surface
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
    if pnt(3) <= 0.0395 && pnt(3) >= 0.0205
        if sign(penet_temp) ~= sign(penet)
            penet = penet_temp
        end
    end
    normal = normal1;
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

%% plot
% normal: obj_f_vn !!
trimesh(obj_f_v, obj_v(:, 1), obj_v(:, 2), obj_v(:, 3))
hold on;
plot3(pnt(1), pnt(2), pnt(3),'.', 'Color', 'b')
normal2 = normal * 0.01;
quiver3(min_pnt(1), min_pnt(2), min_pnt(3), normal2(1), normal2(2), normal2(3))
