%% Preliminaries

clc;
close all;

load('../results/18/OutputModel2.mat')
figpath   = '../figures/';
appenpath = [figpath 'appendix/'];

set(0,'defaultAxesFontName', 'Times');
set(0,'DefaultTextInterpreter', 'latex')
set(0, 'DefaultAxesFontSize',15)
set(0,'defaultAxesLineStyleOrder','-|--|:', 'defaultLineLineWidth',1.5)
setappdata(0, 'defaultAxesXTickFontSize', 1)
setappdata(0, 'defaultAxesYTickFontSize', 1)
addpath('Routines');

Quant = [0.025 0.160 0.500 0.840  0.975];
M = size(CommonTrends, 3);

%% Extract Common Trends

codes = {'us','de','uk','fr','ca','it','jp'};
Nc = numel(codes);
T = size(CommonTrends, 1);
Nd = size(CommonTrends, 3);

M_bar = squeeze(CommonTrends(:, 1, :));
Pi_bar = squeeze(CommonTrends(:, 2, :));
Ts_bar = squeeze(CommonTrends(:, 3, :));
Cy_bar = squeeze(CommonTrends(:, 4, :));
Rshort_bar = M_bar - Cy_bar;
Rlong_bar = Rshort_bar + Ts_bar;

%% Extract Country-Specific Idiosyncratic Components

base = 4;
idx_rs = base + (1:Nc);
idx_pi = base + Nc + (1:Nc);
idx_ts = base + 2*Nc + (1:Nc);

for i = 1:Nc
    code = codes{i};
    
    Rs_idio_i = squeeze(CommonTrends(:, idx_rs(i), :));
    Pi_idio_i = squeeze(CommonTrends(:, idx_pi(i), :));
    Ts_idio_i = squeeze(CommonTrends(:, idx_ts(i), :));
    
    eval(sprintf('Rshort_bar_%s_idio = Rs_idio_i;', code));
    eval(sprintf('Pi_bar_%s_idio = Pi_idio_i;', code));
    eval(sprintf('Ts_bar_%s_idio = Ts_idio_i;', code));
    eval(sprintf('Rlong_bar_%s_idio = Rs_idio_i + Ts_idio_i;', code));
end

%% Calculate Country-Specific Trends

for i = 1:Nc
    code = codes{i};
    
    w_rs = repmat(transpose(squeeze(CC(i, 1, :))), T, 1);
    w_pi = repmat(transpose(squeeze(CC(i, 2, :))), T, 1);
    
    Rs_idio_i = eval(sprintf('Rshort_bar_%s_idio', code));
    Pi_idio_i = eval(sprintf('Pi_bar_%s_idio', code));
    Ts_idio_i = eval(sprintf('Ts_bar_%s_idio', code));
    
    Rshort_i = w_rs .* Rshort_bar + Rs_idio_i;
    Pi_i = w_pi .* Pi_bar + Pi_idio_i;
    Ts_i = Ts_bar + Ts_idio_i;
    Rlong_i = Rlong_bar + Rs_idio_i + Ts_idio_i;
    
    eval(sprintf('Rshort_bar_%s = Rshort_i;', code));
    eval(sprintf('Pi_bar_%s = Pi_i;', code));
    eval(sprintf('Ts_bar_%s = Ts_i;', code));
    eval(sprintf('Rlong_bar_%s = Rlong_i;', code));
end



%% Sort Common Trends

sRshort_bar = sort(Rshort_bar, 2);
sPi_bar = sort(Pi_bar, 2);
sTs_bar = sort(Ts_bar, 2);
sCy_bar = sort(Cy_bar, 2);
sM_bar = sort(M_bar, 2);
sRlong_bar = sort(Rlong_bar, 2);

%% Sort Country-Specific Trends

for k = 1:Nc
    c = codes{k};
    eval(sprintf('sRshort_bar_%s = sort(Rshort_bar_%s, 2);', c, c));
    eval(sprintf('sRlong_bar_%s = sort(Rlong_bar_%s, 2);', c, c));
    eval(sprintf('sPi_bar_%s = sort(Pi_bar_%s, 2);', c, c));
    eval(sprintf('sTs_bar_%s = sort(Ts_bar_%s, 2);', c, c));
    eval(sprintf('sRshort_bar_%s_idio = sort(Rshort_bar_%s_idio, 2);', c, c));
end

%% Calculate Quantiles of Common Trends

M = size(sRshort_bar, 2);
qInd = max(1, min(M, ceil(Quant(:)' .* M)));

qRshort_bar = sRshort_bar(:, qInd);
qPi_bar = sPi_bar(:, qInd);
qTs_bar = sTs_bar(:, qInd);
qCy_bar = sCy_bar(:, qInd);
qM_bar = sM_bar(:, qInd);
qRlong_bar = sRlong_bar(:, qInd);

%% Calculate Quantiles of Country-Specific Trends

for k = 1:Nc
    c = codes{k};
    eval(sprintf('qRshort_bar_%s = sRshort_bar_%s(:, qInd);', c, c));
    eval(sprintf('qRlong_bar_%s = sRlong_bar_%s(:, qInd);', c, c));
    eval(sprintf('qPi_bar_%s = sPi_bar_%s(:, qInd);', c, c));
    eval(sprintf('qTs_bar_%s = sTs_bar_%s(:, qInd);', c, c));
    eval(sprintf('qRshort_bar_%s_idio = sRshort_bar_%s_idio(:, qInd);', c, c));
end
%
%%

[M, idx] = max(qRshort_bar(:,3));
disp("max");
disp(M);
disp(idx + 1870);

[M, idx] = max(qRshort_bar(121:155,3));
disp("max 1990-");
disp(M);
disp(idx);

[M, idx] = min(qRshort_bar(120:150,3));
disp("min 1990-");
disp(M);
disp(idx + 1870 + 120);

disp("change 1990-2020 r*")
disp(qRshort_bar(121,3));
disp(qRshort_bar(151,3));

disp("change 2020-2024 r*")
disp(qRshort_bar(151,3));
disp(qRshort_bar(155,3));

disp("change 1990-2020 m")
disp(qM_bar(121,3));
disp(qM_bar(151,3));

disp("change 2020-2024 m")
disp(qM_bar(151,3));
disp(qM_bar(155,3));

t_start = find(Year == 2024);
disp(t_start);