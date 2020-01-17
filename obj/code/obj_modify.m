
obj = readObj('../obj_file/coarse_bolt_v3_size_reduced.obj');

ps = obj.v;
ns = obj.vn;
face = obj.f.v;
pnt_num = size(ps, 1);
face_num = size(face, 1);

%% find duplicate points
duplicate_pnt = [];
for i = 1:pnt_num-1
    for j = i+1:pnt_num
        if norm(ps(i,:)-ps(j,:)) < 0.0001
            duplicate_pnt = [duplicate_pnt ; [i,j]];
        end
    end
end

%% duplicate point modification
duplicate_pnt_bef = duplicate_pnt;
for i=1:size(duplicate_pnt,1)
    for j = i+1 : size(duplicate_pnt,1)
        if (duplicate_pnt(i,2) == duplicate_pnt(j,1))
            duplicate_pnt(j,1) = duplicate_pnt(i,1);
        end
        if (duplicate_pnt(i,2) == duplicate_pnt(j,2))
            duplicate_pnt(j,2) = duplicate_pnt(j,1);
            duplicate_pnt(j,1) = duplicate_pnt(i,1);
        end
    end
end
idx_dup= [];
for i = 1:size(duplicate_pnt,1)
    if (duplicate_pnt(i,1)~=duplicate_pnt(i,2))
        idx_dup= [idx_dup i];
    end
end
duplicate_pnt = duplicate_pnt(idx_dup,:);

%% make new_ind
new_ind = [];
compensation = zeros(pnt_num,1);

% initialization
new_ind = 1:1:pnt_num;

for i = 1:size(duplicate_pnt,1)
    for j = (duplicate_pnt(i,2)+1):pnt_num
%         new_ind(j) = new_ind(j) - 1;
        compensation(j) = compensation(j) + 1;
    end
end
for j = 1:pnt_num
    new_ind(j) = new_ind(j) - compensation(j);
end

for i=1:size(duplicate_pnt,1)
    new_ind(duplicate_pnt(i,2)) = new_ind(duplicate_pnt(i,1));
end

%% find erase index
new_face=[];
face2 = face;
for i = 1:face_num
    for j=1:3
        face2(i,j) = new_ind(face(i,j));
    end
    if (face2(i,1)==face2(i,2) || face2(i,1)==face2(i,3) || face2(i,2) == face2(i,3))
        continue;
    end
    new_face = [new_face; face2(i,:)];
end

%%
new_pnt = zeros(max(new_ind),3);
new_normal = zeros(max(new_ind),3);
for i = 1:pnt_num
    new_pnt(new_ind(i),:) = ps(i,:);
    new_normal(new_ind(i),:) = ns(i,:);
end
"done"

%%
face_nut = new_face.';
ps_nut = new_pnt.';
ns_nut = new_normal.';

N = size(face_nut,2);
Gst = zeros(2,3*N);
for i = 1:N
    Gst(1,3*(i-1)+1) = face_nut(1,i);
    Gst(1,3*(i-1)+2) = face_nut(1,i);
    Gst(1,3*(i-1)+3) = face_nut(2,i);
    Gst(2,3*(i-1)+1) = face_nut(2,i);
    Gst(2,3*(i-1)+2) = face_nut(3,i);
    Gst(2,3*(i-1)+3) = face_nut(3,i);
end
% Gst = [Gst index];
Gst = unique(sort(Gst).','rows','stable').';
G = graph(Gst(1,:),Gst(2,:));
plot(G)
% figure(2);
% hold on;
% plot(G,'XData',ps_nut(1,:),'YData',ps_nut(2,:),'ZData',ps_nut(3,:))