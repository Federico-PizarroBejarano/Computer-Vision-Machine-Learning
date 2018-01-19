function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);      

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

ny = zeros(m, size(a3, 2));
for i = 1:m,
  ny(i, y(i)) = 1;
end;

J = (1/m)*sum(sum(-ny.*log(a3) - (1-ny).*log(1-a3)));

J = J + (lambda/(2*m))*(sum(sum(Theta1(:, [2: size(Theta1, 2)]).^2)) + sum(sum(Theta2(:, [2: size(Theta2, 2)]).^2)));

d3 = a3-ny;
d2 = d3*Theta2(:, [2:end]) .* sigmoidGradient(z2);

regTheta1 = Theta1;
regTheta1(:, 1) = zeros(size(Theta1, 1), 1);

regTheta2 = Theta2;
regTheta2(:, 1) = zeros(size(Theta2, 1), 1);

D2 = (1/m)*(d3'*a2) + (lambda/m)*(regTheta2);
D1 = (1/m)*(d2'*X) + (lambda/m)*(regTheta1);

grad = [D1(:) ; D2(:)];


end
