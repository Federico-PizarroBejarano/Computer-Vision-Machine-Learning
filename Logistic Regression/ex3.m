clear ; close all;

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 36;            
lambda = 0; 
iters = 100;                       

fprintf('Lambda: %f\n', lambda)
fprintf('Iterations: %f\n', iters)

load FntX144x4.mat;
X = FntX;
clear FntX;
X = [X];

load Fnty144x4.mat;
y = Fnty;
clear Fnty;

disp("training input loaded");

tot = [X, y];
totr = tot(randperm(size(tot,1)),:);
X = totr(:, 1:(end-1));
y = totr(:, end);

t0 = clock ();

% Some useful variables
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

options = optimset('GradObj', 'on', 'MaxIter', iters);

for i=1:(num_labels),
  theta = zeros(n+1, 1);
  [temp] = ...
    fmincg (@(t)(lrCostFunction(t, [ones(m, 1) X], (y == i), lambda)), ...
    theta, options);
   all_theta(i, :) = temp;
end;

t1 = clock();
dt = etime(t1, t0);

%% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

clear X;
clear y;

load FntXCV144x4.mat;
XCV = FntXCV;
clear FntXCV;

load FntyCV144x4.mat;
yCV = FntyCV;
clear FntyCV;

disp("CV input loaded");

pred = predictOneVsAll(all_theta, [XCV]);

fprintf('CV Set Accuracy: %f\n', mean(double(pred == yCV)) * 100);

clear XCV;
clear yCV;

load FntXtest144x4.mat;
Xtest = FntXtest;
clear FntXtest;

load Fntytest144x4.mat;
ytest = Fntytest;
clear Fntytest;

disp("test input loaded");

pred = predictOneVsAll(all_theta, [Xtest]);

fprintf('Test Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
disp(" ")

fprintf('Time Elapsed: %f\n', dt);