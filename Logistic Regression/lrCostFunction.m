function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

J = (1/m)*sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta))) + (lambda/(2*m))*sum(theta([2:1:length(theta)], :).^2);

grad0 = (1/m)*(sum(sigmoid(X*theta) - y))';
gradj = (1/m)*(sum((sigmoid(X*theta) - y).*X(:, [2:1:size(X, 2)])))' + (lambda/m)*theta([2:1:length(theta)], :);

grad = [grad0; gradj];

grad = grad(:);

end
