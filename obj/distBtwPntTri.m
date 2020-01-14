function [pnt, dist, geo, s, t] = distBtwPntTri(tri, P)
%DISTBTWPNTTRI Summary of this function goes here
%   Detailed explanation goes here
% scale = 10000;
% tri = tri * scale;
% P = P * scale;

B = tri(1, :);
E0 = tri(2, :) - tri(1, :);
E1 = tri(3, :) - tri(1, :);

a = dot(E0, E0);
b = dot(E0, E1);
c = dot(E1, E1);
d = dot(E0, B-P);
e = dot(E1, B-P);
f = dot(B-P, B-P);


s = b * e - c * d;
t = b * d - a * e;
det = a * c - b * b;
if (s + t <= det)
    if (s < 0)
        if (t < 0)
            % region 4
            if (d < 0)
                t = 0;
                if (-d >= a)
                    s = 1;
                else
                    s = -d/a;
                end
            else
                s = 0;
                if ( e >= 0)
                    t = 0;
                elseif (-e >= c)
                    t = 1;
                else
                    t = -e / c;
                end
            end
        else
            % region 3
            s = 0;
            if (e >= 0)
                t = 0;
            elseif (-e >= c)
                t = 1;
            else
                t = -e / c;
            end
        end
    elseif ( t < 0)
        % region 5
        t = 0;
        if (d >= 0)
            s = 0;
        elseif (-d >= a)
            s = 1;
        else
            s = -d/a;
        end
    else
        % region 0
        s = s / det;
        t = t / det;
    end
else
    if (s < 0)
        % region 2
        tmp0 = b + d;
        tmp1 = c + e;
        if (tmp1 > tmp0)
            numer = tmp1 - tmp0;
            denom = a - 2*b + c;
            if (numer >= denom)
                s = 1;
            else
                s = numer / denom;
            end
            t = 1 - s;
        else
            s = 0;
            if (tmp1 <= 0)
                t = 1;
            elseif (e >= 0)
                t = 0;
            else
                t = -e / c;
            end
        end
    elseif (t < 0)
        % region 6
        tmp0 = b + e;
        tmp1 = a + d;
        if (tmp1 > tmp0)
            numer = tmp1 - tmp0;
            denom = a - 2*b + c;
            if (numer >= denom)
                t = 1;
            else
                t = numer / denom;
            end
            s = 1 - t;
        else
            t = 0;
            if (tmp1 <= 0)
                s = 1;
            elseif (d >= 0)
                s = 0;
            else
                s = -d/a;
            end
        end
    else
        % region 1
        numer = (c+e) - (b+d);
        if (numer <= 0)
            s = 0;
        else
            denom = a - 2*b + c;
            if (numer >= denom)
                s = 1;
            else
                s = numer / denom;
            end
        end
        t = 1 - s;
    end    
end
pnt = B + s * E0 + t * E1;
dist = norm(pnt - P);

if (s == 0 || s == 1) && (t ==0 || t == 1)
    geo = 0;
elseif (s + t == 1) || (s == 0) || (t == 0)
    geo = 1;
else
    geo = 2;
end

% pnt = pnt / scale;
% dist = dist / scale;
end

