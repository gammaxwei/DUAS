function feature_slc = DUAS(X, options, flag)
% DUAS
% Input:
%   X       : 1 x V cell, each X{v} is n x d_v
%   options : struct with fields alpha and beta
%   flag    : whether to draw convergence curve (0/1)
%
% Output:
%   feature_slc : 1 x V cell, where feature_slc{v} contains the feature
%                 indices of view v ranked in descending importance

if nargin < 3
    flag = 0;
end

num_view = length(X);
d_ = 0;
d = zeros(1, num_view);
m_i = zeros(1, num_view);

for i = 1:num_view
    d(i) = size(X{i}, 2);
    n = size(X{1}, 1);
    m_i(i) = ceil(0.02 * d(i));   % feature anchors
    d_ = d_ + m_i(i);
end

if n < 10000
    n_i = ceil(0.1 * n);          % sample anchors
else
    n_i = ceil(0.01 * n);
end

options.V = num_view;
options.r1 = n_i;
options.k = d_;
options.n = n;

V = num_view;

% -------------- Initialize variables -------------- %
M = cell(1, V);
P = cell(1, V);
W = cell(1, V);
for v = 1:V
    P{v} = rand(m_i(v), d(v));
    M{v} = rand(n_i, m_i(v));
    W{v} = rand(d_, m_i(v));
end

Z = zeros(options.n, options.r1, V);
for v = 1:V
    Z(:, :, v) = rand(options.n, options.r1);
end

A = rand(options.n, options.k);
Q = rand(V, options.k);
H = rand(options.r1, options.k);

% -------------- Optimization -------------- %
max_iter = 40;
tol = 1e-3;
obj = zeros(max_iter, 1);

for iter = 1:max_iter
    % Update 
    H = updateH(W, M, Z, A, Q, options);
    A = updateA(Z, Q, H);
    Q = updateQ(Z, H, A);
    Z = updateZ(X, A, Q, M, P, H);
    M = updateM(W, H, M, X, Z, P, options);
    P = updatePi(X, Z, M);
    W = updateW(M, H);

    obj(iter) = obj_fun(X, P, Z, M, W, A, H, Q, options);

    if iter > 1
        rel_change = abs((obj(iter) - obj(iter - 1)) / max(abs(obj(iter - 1)), eps));
        if rel_change < tol
            obj = obj(1:iter);
            break;
        end
    end
end

% -------------- Feature selection result -------------- %
feature_slc = cell(1, V);
for v = 1:V
    C = P{v}' * M{v}';
    B = P{v}' * W{v}';
    score_v = sum(C, 2) + sum(B, 2);
    [~, feature_slc{v}] = sort(score_v, 'descend');
end

if flag == 1
    picture(obj);
end

end

%% ---------------- objective function ----------------
function f = obj_fun(X, P, Z, M, W, A, H, Q, options)
V = length(X);
f1 = 0;
f2 = 0;
f3 = 0;
for v = 1:V
    f1 = f1 + norm(X{v} - Z(:, :, v) * M{v} * P{v}, 'fro')^2 ...
            + options.alpha * norm(M{v}, 'fro')^2;
    f2 = f2 + options.beta * norm(H * W{v} - M{v}, 'fro')^2;
    f3 = f3 + norm(Z(:, :, v) - A * diag(Q(v, :)) * H', 'fro')^2;
end
f = f1 + f2 + f3;
end

%% ---------------- update A ----------------
function A = updateA(Z, Q, H)
tensor_size = size(Z);
Z_mode_1 = reshape(Z, tensor_size(1), []);
QH = khatrirao(Q, H);
C = Z_mode_1 * QH;
[Uv, ~, Vv] = svd(C, 'econ');
A = Uv * Vv';
end

%% ---------------- update H ----------------
function H = updateH(W, M, Z, A, Q, options)
tensor_size = size(Z);
Z_mode_2 = reshape(permute(Z, [2 1 3]), tensor_size(2), []);
QA = khatrirao(Q, A);
MW = zeros(options.r1, options.k);
V = length(M);
for v = 1:V
    MW = MW + M{v} * W{v}';
end
C = options.beta * MW + Z_mode_2 * QA;
[Uv, ~, Vv] = svd(C, 'econ');
H = Uv * Vv';
end

%% ---------------- update Q ----------------
function Q = updateQ(Z, H, A)
tensor_size = size(Z);
Z_mode_3 = reshape(permute(Z, [3 1 2]), tensor_size(3), []);
HA = khatrirao(H, A);
Q = Z_mode_3 * HA * pinv(HA' * HA);
end

%% ---------------- update Z ----------------
function Z = updateZ(X, A, Q, M, P, H)
V = length(X);
for v = 1:V
    C = X{v} * P{v}' * M{v}' + A * diag(Q(v, :)) * H';
    [Uv, ~, Vv] = svd(C, 'econ');
    Z(:, :, v) = Uv * Vv';
end
end

%% ---------------- update P ----------------
function P = updatePi(X, Z, M)
V = length(X);
P = cell(1, V);
for v = 1:V
    C = X{v}' * Z(:, :, v) * M{v};
    [Uv, ~, Vv] = svd(C, 'econ');
    P{v} = Vv * Uv';
end
end

%% ---------------- update M ----------------
function M = updateM(W, H, M, X, Z, P, options)
opts = optimoptions('quadprog', 'Display', 'off');
warning('off', 'optim:quadprog:HessianNotSym');
V = length(X);
for v = 1:V
    Zv = Z(:, :, v)';
    I = eye(size(P{v}, 1));
    O = 2 * (P{v}*P{v}' + options.beta + options.alpha);
    for j = 1:options.r1
        s = -2 * (options.beta * W{v}' * H(j, :)' + P{v} * X{v}' * Zv(j, :)');
        Aeq = ones(1, length(s));
        beq = 1;
        lb = zeros(length(s), 1);
        U = quadprog(O, s, [], [], Aeq, beq, lb, [], [], opts);
        M{v}(j, :) = U';
    end
end
warning('on', 'optim:quadprog:HessianNotSym');
end

%% ---------------- update W ----------------
function W = updateW(M, H)
V = length(M);
W = cell(1, V);
for v = 1:V
    W{v} = pinv(H' * H) * (H' * M{v});
end
end

%% ---------------- draw picture ----------------
function picture(obj)
figure('Color', [1 1 1]);
plot(obj, '-*', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel('Iteration number');
ylabel('Objective function value');
box on;
grid on;
end
