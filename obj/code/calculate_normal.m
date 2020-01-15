clear
addpath('../data')
load('../data/obj_data.mat')

%%
new_normal = zeros(size(new_face, 1), 3);

idx = 1;
% calculate normal of first surface
vec1 = new_pnt(new_face(idx, 2), :) - new_pnt(new_face(idx, 1), :);
vec2 = new_pnt(new_face(idx, 3), :) - new_pnt(new_face(idx, 1), :);
normal = cross(vec1, vec2) / norm(cross(vec1, vec2));
new_normal(1, :) = normal;

%%
a1 = find(new_face(:, 1) == idx);
a2 = find(new_face(:, 2) == idx);
a3 = find(new_face(:, 3) == idx);

set1 = intersect(a1, a2);
set2 = intersect(a2, a3);
set3 = intersect(a1, a3);

adj_face = [];
adj_face = [adj_face; set1; set2; set3];
adj_type = [adj_type; ones(numel(set1)); ones(numel(set2)); ones(numel(set3))];

for l = 1:numel(adj_face)
    findRealNorm(new_normal, new_face, new_point, normal, pnts, idx, ...
        adj_face(l), new_face(1, :), adj_type(l));
end
