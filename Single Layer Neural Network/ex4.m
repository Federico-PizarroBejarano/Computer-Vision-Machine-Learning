clear
close all

ils  = 400;  % Digits
hls = 25;   % hidden units
nl = 36;     % labels     
lambda = 0;  
iters = 10;

fprintf('Hidden units: %f\n', hls)
fprintf('Number of labels: %f\n', nl)
fprintf('Lambda: %f\n', lambda)
fprintf('Iterations: %f\n', iters)

%Training set............
load FntX144x4.mat;
X = FntX;

load Fnty144x4.mat;
y = Fnty;

disp("training input loaded");

tot = [X, y];
totr = tot(randperm(size(tot,1)),:);
X = totr(:, 1:(end-1));
y = totr(:, end);

m = size(X, 1);

initial_Theta1 = randInitializeWeights(ils, hls);
initial_Theta2 = randInitializeWeights(hls, nl);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


options = optimset('MaxIter', iters);

t0 = clock();


costFunction = @(p) nnCostFunction(p, ...
                                   ils, ...
                                   hls, ...
                                   nl, X, y, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

t1 = clock();

Theta1 = reshape(nn_params(1:hls * (ils + 1)), ...
                 hls, (ils + 1));

Theta2 = reshape(nn_params((1 + (hls * (ils + 1))):end), ...
                 nl, (hls + 1)); 
     
pred = predict(Theta1, Theta2, X);

disp(" ")
fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%Test set..............
load FntXCV144x4.mat;
XCV = FntXCV;

load FntyCV144x4.mat;
yCV = FntyCV;


%CV set accuracy

pred = predict(Theta1, Theta2, XCV);

fprintf('CV Set Accuracy: %f\n', mean(double(pred == yCV)) * 100);
disp(" ")

%Test set..............
load FntXtest144x4.mat;
Xtest = FntXtest;

load Fntytest144x4.mat;
ytest = Fntytest;

%test set accuracy
pred = predict(Theta1, Theta2, Xtest);

fprintf('Test Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
disp(" ")

disp(etime(t1, t0));