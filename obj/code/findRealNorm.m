function [outputArg1,outputArg2] = findRealNorm(new_normal, new_face, ...
    new_pnt, normal, pnts, idx, prev_surf_ind type)
%FINDREALNORM Summary of this function goes here
%   Detailed explanation goes here
EPS = 0.00001;
vec1 = new_pnt(new_face(idx, 1), :) - new_pnt(new_face(idx, 2), :);
vec2 = new_pnt(new_face(idx, 1), :) - new_pnt(new_face(idx, 3), :);
normal2 = cross(vec1, vec2) / norm(cross(vec1, vec2);


end

