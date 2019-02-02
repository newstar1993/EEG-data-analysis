function X = Gaussian_copy(y,std_y)
y_copy = repmat(y,100,1);
A = (1/100)*diag(std_y);
Noise = [];
[m,n] = size(y);
for i = 1: size(y_copy,1)
    Noise = [Noise;mvnrnd(zeros(n,1),A)];
end
zero_mat = zeros(m,n);
X = [y ;y_copy] +[zero_mat; Noise];
