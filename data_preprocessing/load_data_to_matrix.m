
%% load the mean_fcz of correct and incorrect data for each subject
% use data augmentation and filtered data with outliers
% to generate 5 datasets for training in total
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mat = dir('*.mat');
clear mean_FCZ;
mean_FCZ = [];
cor_label = [];
std_FCZ = [];
clear ID;
markers_1 =[];
markers_2 =[];
for q = 1:(length(mat)/2)
    load(mat(2*q-1).name);
    mean_temp = mean(FCz);
    [o_temp,o_temp2,ms2_t1]  = isStimulus(Markers);
    std_temp = std(FCz);
    load(mat(2*q).name);
    mean_temp = [mean_temp,mean(FCz)];
    std_temp = [std_temp,std(FCz)];
    [o_temp_2,o_temp2_2,ms2_t2]  = isStimulus(Markers);
    markers_1 = [o_temp,o_temp_2];
    markers_2 = [o_temp2,o_temp2_2];
    mss = [ms2_t1,ms2_t2];
    avg_markers_1(q) = mean(markers_1);
    avg_markers_2(q) = mean(markers_2);
    % avg_markers_1(q) = mean(mss);
    mean_FCZ = [mean_FCZ; mean_temp];
    std_FCZ = [std_FCZ; std_temp];
end
final_label = zeros(91,1);

for i =1: size(GroupIDs,1)
    
    final_label(i)=GroupIDs(i);
    ID(i)=i;
    
end

data = [mean_FCZ,avg_markers_1',avg_markers_2'];
X_temp = region_copy(mean_FCZ, 50,ID',final_label,avg_markers_1',avg_markers_2');
final_data = X_temp;
data_original = [final_label,mean_FCZ,avg_markers_1',avg_markers_2',ID'];
std_data = std(data);
X = Gaussian_copy(data,std_data);

csvwrite('training_data_aug_50.csv',final_data);

csvwrite('original_data.csv',original_data);

%%
clear;
GroupIDs=csvread('GroupIDs.csv');
mat = dir('*.mat');
clear mean_FCZ;
FCz_total = [];
label = [];
markers_1 =[];
markers_2 =[];
patients_ID=[];
mean_FCz =[];

for q = 1:length(mat)/2
    load(mat(2*q-1).name);  
    FCz = draw_hist_fcz(FCz,q);
    [o_temp,o_temp2,ms2_t1]  = isStimulus(Markers);
    FCz_temp = [];
    FCz_temp = [FCz_temp, mean(FCz)];
    load(mat(2*q).name);
    FCz_temp = [FCz_temp, mean(FCz)];
    mean_FCz = [mean_FCz; FCz_temp];
    [o_temp_2,o_temp2_2,ms2_t2]  = isStimulus(Markers);
    markers_1 = [o_temp,o_temp_2];
    markers_2 = [o_temp2,o_temp2_2];
    avg_markers_1(q) = mean(markers_1);
    avg_markers_2(q) = mean(markers_2);

end

label = GroupIDs;
patients_ID = [1:91];
FCz_total = [patients_ID', mean_FCz,avg_markers_1',avg_markers_2',label];
data = [mean_FCz,avg_markers_1',avg_markers_2'];
std_data = std(data);
X = Gaussian_copy(data,std_data);
final_data = [repmat(patients_ID',101,1),X, repmat(label,101,1)];

csvwrite('filtered_data_mean_fcz',final_data);

csvwrite('filtered_data.csv', FCz_total)
