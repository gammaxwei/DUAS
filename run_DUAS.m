clear;
data = load("3sources.mat");
X = data.X;

for v = 1:length(X)
    X{v} = mapminmax(X{v}, 0, 1);
end

alpha = [0.001,0.005,0.01,0.05,0.1,0.5,1];
beta  = [0.001,0.01,0.1,1,10,100,1000];
flag = 0;  % When set to 1, the convergence plot is generated.

for i = 1:length(alpha)
    for j = 1:length(beta)
        options = [];
        options.alpha = alpha(i);
        options.beta = beta(j);

        fprintf('alpha = %.4f, beta = %.4f\n', alpha(i), beta(j));
        feature_slc = DUAS(X, options, flag);
    end
end