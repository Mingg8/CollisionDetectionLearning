clear
input = [];
output = [];

%%
[i, o] = data_postprocessing('data_final.mat');
input = [input; i];
output = [output; o];

save('final_data.mat', 'input', 'output')



%%
% plus_pnt_final = [];
% minus_pnt_final = [];
% 
% for i = 1:size(input_final, 1)
%     if (output_final(i) < 0)
%         minus_pnt_final = [minus_pnt_final; input_final(i, :)];
%     end
% end
% 
% %%
% plot3(minus_pnt_final(:, 1), minus_pnt_final(:, 2), minus_pnt_final(:, 3), '.');
% axis equal