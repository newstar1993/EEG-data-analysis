function [o,m,k]=isStimulus(Markers);
%n=length(Markers);
n=size(Markers,1);
o=zeros(1,n);
m=zeros(1,n);
k=zeros(1,n);
for i=1:n
    if strcmp(Markers(i,1).Type,'Stimulus')
        o(i)=1;
    end
    
    if strcmp(Markers(i,2).Type,'Stimulus')
        m(i)=1;
    end
     if strcmp(Markers(i).Description,'S 1')
        k(i)=1;
    end
   % fprintf('%d\n',i)
end
