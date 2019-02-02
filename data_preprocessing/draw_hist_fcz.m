function fcz_copy = draw_hist_fcz(fcz,i)
figure(i);
distance = pdist2(fcz, mean(fcz));
f = boxplot(distance);
A =(['Box plot of incorrect response of patient',num2str(i)]);
title(A);
h = flipud(findobj(gcf,'tag','Outliers'));
ydata = get(h,'YData');
if length(ydata) <= round(0.2*size(fcz,1))
    m = distance == ydata;
    index = find(any(m == 1, 2)==1);
    %index = find(distance == ydata);
else
    distance_new = ydata((end - round(0.2*size(fcz,1))+1): end);
    m = distance == ydata;
    index = find(any(m == 1, 2)==1);
   % index = find(distance_new == ydata);
end
fcz_copy = fcz;
fcz_copy(index,:) =[];