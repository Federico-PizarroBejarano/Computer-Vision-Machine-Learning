function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);
 
all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];

options = optimset('GradObj', 'on', 'MaxIter', 200);

for i=1:(num_labels),
  theta = zeros(n+1, 1);
  [temp] = ...
    fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
    theta, options);
   all_theta(i, :) = temp;
end;

end;
