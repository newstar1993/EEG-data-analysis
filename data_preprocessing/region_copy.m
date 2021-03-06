function data_end = region_copy(x,locate,Id,y,avg_1,avg_2)
data_end = [];
for i = 1: size(x,1)
    data = x(i,:);
    label = y(i,:);
    id = Id(i,:);
    avg_1_temp = avg_1(i,:);
    avg_2_temp = avg_2(i,:);
    data_1 = data(:,(locate-2):(locate+48));
    data_2 = data(:,(locate-1):(locate+49));
    data_3 = data(:,(locate):(locate+50));
    data_4 = data(:,(locate+1):(locate+51));
    data_5 = data(:,(locate+2):(locate+52));
    data_all = [ data_1;data_2;data_3;data_4;data_5];
    id_temp = [id;id;id;id;id];
    avg1 = [avg_1_temp;avg_1_temp;avg_1_temp;avg_1_temp;avg_1_temp];
    avg2 = [avg_2_temp;avg_2_temp;avg_2_temp;avg_2_temp;avg_2_temp];
    label_temp = [label;label;label;label;label];
    data_temp = [ id_temp, data_all, avg1,avg2, label_temp];
    data_end = [data_end;data_temp];
end
