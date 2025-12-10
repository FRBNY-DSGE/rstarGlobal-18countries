% This file replicates the plots for the "Baseline" model. See figures
% 1, 2, 3, and 4, 5, A1, A2.

%% Plot preliminaries: load, sort, and get quantiles

close all;

addpath('Routines')
load('../results/18/OutputModel1.mat')

figpath   = '../figures/';
appenpath = [figpath 'appendix/'];

Quant = [0.025 0.160 0.500 0.840  0.975];
M = size(CommonTrends, 3);

set(0,'defaultAxesFontName', 'Times');
set(0,'DefaultTextInterpreter', 'latex')
set(0, 'DefaultAxesFontSize',15)
set(0,'defaultAxesLineStyleOrder','-|--|:', 'defaultLineLineWidth',1.5)
setappdata(0, 'defaultAxesXTickFontSize', 1)
setappdata(0, 'defaultAxesYTickFontSize', 1)


% ------------------------- Trends ---------------------------------------

codes = {'us','de','uk','fr','ca','it','jp','au','be','fi','ie','nl','no','ch','se','es','pt', 'dk'};
%codes = {'us', 'de', 'uk', 'fr', 'ca','it','jp', 'au'};
Nc    = numel(codes); %how many countries
T     = size(CommonTrends,1);
Nd    = size(CommonTrends,3);   % number of kept draws

Rshort_bar = squeeze(CommonTrends(:,1,:));
Pi_bar     = squeeze(CommonTrends(:,2,:));
Ts_bar     = squeeze(CommonTrends(:,3,:));
Rlong_bar  = Rshort_bar + Ts_bar;

base = 3;
idx_rs = base + (1:Nc);
idx_pi = base + Nc + (1:Nc);
idx_ts = base + 2*Nc + (1:Nc);
for i = 1:Nc
    code = codes{i};

    Rs_idio_i = squeeze(CommonTrends(:, idx_rs(i), :));
    Pi_idio_i = squeeze(CommonTrends(:, idx_pi(i), :));
    Ts_idio_i = squeeze(CommonTrends(:, idx_ts(i), :));

    % World loadings from C for country i
    w_rs = (squeeze(CC(i,1,:))).';   % rs_wrd loading
    w_pi = (squeeze(CC(i,2,:))).';   % pi_wrd loading

    % Country-level trends (T x Nd)
    Rshort_i = Rshort_bar .* repmat(w_rs, T, 1) + Rs_idio_i;
    Pi_i     = Pi_bar     .* repmat(w_pi, T, 1) + Pi_idio_i;
    Ts_i     = Ts_bar + Ts_idio_i;                          % ts_wrd loads directly in spec
    Rlong_i  = Rlong_bar + Rs_idio_i + Ts_idio_i;

    % Keep legacy variable names (assign into current workspace)
    eval(sprintf('Rshort_bar_%s_idio = Rs_idio_i;', code));
    eval(sprintf('Pi_bar_%s_idio     = Pi_idio_i;', code));
    eval(sprintf('Ts_bar_%s_idio     = Ts_idio_i;', code));
    eval(sprintf('Rlong_bar_%s_idio  = Rs_idio_i + Ts_idio_i;', code));

    eval(sprintf('Rshort_bar_%s = Rshort_i;', code));
    eval(sprintf('Pi_bar_%s     = Pi_i;',     code));
    eval(sprintf('Ts_bar_%s     = Ts_i;',     code));
    eval(sprintf('Rlong_bar_%s  = Rlong_i;',  code));
end

% --------------------------- Sorted trends -------------------------------

sRshort_bar     = sort(Rshort_bar,2);
sPi_bar         = sort(Pi_bar,2);
sTs_bar         = sort(Ts_bar,2);
sRlong_bar      = sort(Rlong_bar,2);

for k = 1:Nc
    c = codes{k};
    eval(sprintf('sRshort_bar_%s      = sort(Rshort_bar_%s,2);',      c, c));
    eval(sprintf('sRlong_bar_%s       = sort(Rlong_bar_%s,2);',       c, c));
    eval(sprintf('sPi_bar_%s          = sort(Pi_bar_%s,2);',          c, c));
    eval(sprintf('sTs_bar_%s          = sort(Ts_bar_%s,2);',          c, c));
    eval(sprintf('sRshort_bar_%s_idio = sort(Rshort_bar_%s_idio,2);', c, c));
end

% === Quantile indices (same logic as before, but safe) ===
M    = size(sRshort_bar, 2);                   % # of draws
qInd = max(1, min(M, ceil(Quant(:)'.*M)));     % 1 x numQuant

qRshort_bar = sRshort_bar(:, qInd);
qPi_bar     = sPi_bar(:,     qInd);
qTs_bar     = sTs_bar(:,     qInd);
qRlong_bar  = sRlong_bar(:,  qInd);


for k = 1:numel(codes)
    c = codes{k};
    eval(sprintf('qRshort_bar_%s      = sRshort_bar_%s(:, qInd);',      c, c));
    eval(sprintf('qRlong_bar_%s       = sRlong_bar_%s(:,  qInd);',      c, c));
    eval(sprintf('qPi_bar_%s          = sPi_bar_%s(:,     qInd);',      c, c));
    eval(sprintf('qTs_bar_%s          = sTs_bar_%s(:,     qInd);',      c, c));
    eval(sprintf('qRshort_bar_%s_idio = sRshort_bar_%s_idio(:, qInd);', c, c));
end




%% Figure 1: Trends in Global and U.S. Real Rates: 1870-2016, Baseline Model

figure

PlotStatesShaded(Year, qRshort_bar)
hold on
plot(Year, qRshort_bar_us(:, 3), ...
    'k:', 'LineWidth', 1.5);  % r-bar US

axis([1880 2024 -3 6])
xticks(1880:20:2020)
% title('$\bar{r}^w_t$ and $\bar{r}_{US, t}$', 'Interpreter', 'latex')
box on

printpdf(gcf, [figpath 'fig1-Model1_Rshortbar-us.pdf'])


%% Figure 2: Trends in Global Real Rates Under Alternative Priors for the 
% Standard Deviation of Innovations to the Trend and Decadal Moving Averages
% figure
% 
% % Load results using disperse prior for variance of innovations to trends
temp = load('../results/OutputModel1_var01.mat', 'CommonTrends');
CommonTrends_var01 = temp.CommonTrends;
Rshort_bar_var01   = squeeze(CommonTrends_var01(:, 1,:));
sRshort_bar_var01  = sort(Rshort_bar_var01,2);
qRshort_bar_var01  = sRshort_bar_var01(:,ceil(Quant*M));


% Compute real interest rate
rir = [Stir_us-Infl_us, Stir_de-Infl_de, Stir_uk-Infl_uk, ...
    Stir_fr-Infl_fr, Stir_ca-Infl_ca, Stir_it-Infl_it, Stir_jp-Infl_jp];
rir(abs(rir) > 30) = NaN;  % Remove extreme observations

rir_ma = nan(size(rir));
h = 5;  % Centered moving average (plus/minus h years)

for iVar = 1:size(rir,2)
    z_i  = rir(:,iVar);
    rir_ma(:,iVar) = ma_centered(z_i, h);
end

rir_ma_world = nanmean(rir_ma, 2);  % Take cross-sectional average

bands_new(Year, qRshort_bar); hold on;
bands_new(Year, [], qRshort_bar_var01);
bands_new(Year, qRshort_bar);
plot(Year, rir_ma_world, 'LineWidth', 2, 'LineStyle', '-',...
    'Color', 0.75 * [0.9544, 0.0780, 0.1840]);

axis([Year(1) Year(end) -12 12])
xticks(1880:20:2020)

printpdf(gcf, [figpath 'fig2-Model1_var01_Rbar-MA.pdf'])
saveas(gcf, 'figure1.eps', 'epsc');

%% Figure 3a: Trends and Observables for Short-Term Real Rates, Baseline Model
figure
PlotStatesShaded(Year, qRshort_bar);
hold on

codes = {'us','de','uk','fr','ca','it','jp','au','be','fi','ie','nl','no','ch','se','es','pt', 'dk'};
labels = upper(codes);

country_colors = [
    0    0    0;
    0    0    1;
    0    1    1;
    1    1    0;
    1    0    0;
    0    1    0;
    1    0    1;
    1    0.5  0;
    0.5  0    0.5;
    0    0.5  0.5;
    0.5  0.5  0;
    0    0    0.5;
    0.7  0.3  0;
    0.3  0.3  0.3;
    1    0.7  0;
    0.8  0    0.2;
    0    0.7  0.3;
];

ph = gobjects(1, numel(codes));

for k = 1:numel(codes)
    c = codes{k};
    
    real_rate = eval(sprintf('Stir_%s - Infl_%s', c, c));
    
    ph(k) = plot(Year, real_rate, ':', ...
        'Color', country_colors(k,:), ...
        'LineWidth', 1);
    
    eval(sprintf('p_%s = ph(k);', c));
end

axis([Year(1) Year(end) -6 15])
xticks(1880:20:2020)
yticks(-5:5:15)

lgd = legend(ph, labels, ...
    'Interpreter', 'latex', ...
    'Location', 'southoutside', ...
    'FontSize', 9, ...
    'Orientation', 'horizontal', ...
    'NumColumns', 6);

legend boxoff
box on

printpdf(gcf, [figpath 'fig3a-Model1_Rshortbar-obs.pdf'])

%% Figure 3b: Trends and Observables for Short-Term Real Rates, Baseline Model
figure
hold on

ph = gobjects(1, numel(codes));

for k = 1:numel(codes)
    c = codes{k};
    
    qRshort_data = eval(sprintf('qRshort_bar_%s(:,3)', c));
    
    ph(k) = plot(Year, qRshort_data, ':', ...
        'Color', country_colors(k,:), ...
        'LineWidth', 2);
    
    eval(sprintf('p_%s = ph(k);', c));
end

plot(Year, qRshort_bar(:, 3), 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

hline = refline(0);
hline.HandleVisibility = 'off';
hline.Color = 'k';
xticks(1880:20:2020)

xlim([Year(1) Year(end)])
ylim([-3 6])
yticks(-2:2:6)

lgd = legend(ph, labels, ...
    'Interpreter', 'latex', ...
    'Location', 'southoutside', ...
    'FontSize', 9, ...
    'Orientation', 'horizontal', ...
    'NumColumns', 6);

legend boxoff
box on

printpdf(gcf, [figpath 'fig3b-Model1_Rshortbar-countries.pdf'])
saveas(gcf, 'figure3.eps', 'epsc');

%% Compute table 1 main model values
% 
% Table1 = struct;
% 
% Table1(1).Delta_rw = CommonTrends(Year == 2016, 1, :) - CommonTrends(Year == 1980, 1, :);
% Table1(1).Years = [1980 2016];
% 
% Table1(2).Delta_rw = CommonTrends(Year == 2016, 1, :) - CommonTrends(Year == 1990, 1, :);
% Table1(2).Years = [1990 2016];
% 
% Table1(3).Delta_rw = CommonTrends(Year == 2016, 1, :) - CommonTrends(Year == 1997, 1, :);
% Table1(3).Years = [1997 2016];
% 
% clc
% 
% for j = 1:size(Table1, 2)
%     disp('')
%     disp(['-----------------[' num2str(Table1(j).Years) ']------------------'])
%     disp(['Median:' num2str(quantile(Table1(j).Delta_rw, .5))])
%     disp(['90% posterior coverage: ' ...
%         '[' num2str(quantile(Table1(j).Delta_rw, .05)) ', ' num2str(quantile(Table1(j).Delta_rw, .95)) ']'])
% end

% Define analysis periods
periods = [
    1980, 2019;
    2019, 2024
];

% Define the exact same variable as original code
variables = struct();
variables(1).name = 'Delta_rw';
variables(1).index = 1;  % World real rate (first column, same as original)
variables(1).description = 'World Real Rate';

% Initialize results structure
Table1 = struct();
counter = 1;

% Loop over variables and periods
for v = 1:length(variables)
    for p = 1:size(periods, 1)
        start_year = periods(p, 1);
        end_year = periods(p, 2);
        
        % Calculate change over the period (all draws, all countries)
        start_idx = find(Year == start_year, 1);
        end_idx = find(Year == end_year, 1);
        
        if isempty(start_idx) || isempty(end_idx)
            warning('Year %d or %d not found in data', start_year, end_year);
            continue;
        end
        
        % Extract data: (draws x countries)
        start_data = squeeze(CommonTrends(start_idx, variables(v).index, :));
        end_data = squeeze(CommonTrends(end_idx, variables(v).index, :));
        
        % Store results
        Table1(counter).(variables(v).name) = end_data - start_data;
        Table1(counter).Years = [start_year, end_year];
        Table1(counter).Variable = variables(v).description;
        
        counter = counter + 1;
    end
end

% Display results and create formatted table
clc

% Initialize storage for table data
table_data = {};
row_labels = {};

% Process each period
for j = 1:length(Table1)
    period_str = sprintf('%d-%d', Table1(j).Years(1), Table1(j).Years(2));
    data = Table1(j).(variables(1).name);
    
    % Calculate statistics
    median_val = quantile(data, 0.5);
    ci_lower = quantile(data, 0.05);
    ci_upper = quantile(data, 0.95);
    
    % Calculate p-value (two-tailed test that change = 0)
    % P(|change| > 0) = 1 - P(change = 0)
    % For Bayesian posterior: p-value = 2 * min(P(change > 0), P(change < 0))
    prob_positive = mean(data > 0);
    prob_negative = mean(data < 0);
    p_value = 2 * min(prob_positive, prob_negative);
    
    % Store data for table
    table_data{j, 1} = sprintf('%.3f', median_val);
    table_data{j, 2} = sprintf('[%.3f, %.3f]', ci_lower, ci_upper);
    table_data{j, 3} = sprintf('%.3f', p_value);
    row_labels{j} = period_str;
    
    % Also display in console (original format)
    fprintf('\n-----------------[%s]------------------\n', period_str);
    fprintf('Median: %.4f\n', median_val);
    fprintf('90%% posterior coverage: [%.4f, %.4f]\n', ci_lower, ci_upper);
    fprintf('P-value (H0: change = 0): %.4f\n', p_value);
end

% Create and display formatted table
fprintf('\n\n=== FORMATTED TABLE ===\n');
fprintf('%-12s | %-10s | %-20s | %-10s\n', 'Period', 'Median', '90% CI', 'P-value');
fprintf('%s\n', repmat('-', 1, 58));
for j = 1:length(Table1)
    fprintf('%-12s | %-10s | %-20s | %-10s\n', row_labels{j}, table_data{j, 1}, table_data{j, 2}, table_data{j, 3});
end

% Generate LaTeX table
latex_filename = 'world_real_rate_changes.tex';
fid = fopen(latex_filename, 'w');

fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Changes in World Real Rate}\n');
fprintf(fid, '\\label{tab:world_real_rate_changes}\n');
fprintf(fid, '\\begin{tabular}{lccc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'Period & Median & 90\\%% Posterior Interval & P-value \\\\\n');
fprintf(fid, '\\midrule\n');

for j = 1:length(Table1)
    fprintf(fid, '%s & %s & %s & %s \\\\\n', row_labels{j}, table_data{j, 1}, table_data{j, 2}, table_data{j, 3});
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\begin{tablenotes}\n');
fprintf(fid, '\\small\n');
fprintf(fid, '\\item Notes: Changes in world real rate computed from posterior draws.\n');
fprintf(fid, '\\item 90\\%% posterior intervals show the 5th and 95th percentiles.\n');
fprintf(fid, '\\item P-values test the null hypothesis that the change equals zero (two-tailed).\n');
fprintf(fid, '\\end{tablenotes}\n');
fprintf(fid, '\\end{table}\n');

fclose(fid);

fprintf('\nLaTeX table saved to: %s\n', latex_filename);


% 
% 
% %% Figure 4a: Trends and Observables for Inflation, Baseline Model
% 
% figure;
% 
% PlotStatesShaded(Year, qPi_bar); hold on;  % pi-bar world
% p_us = plot(Year, Infl_us, 'k:', 'LineWidth', 1); hold on;
% p_de = plot(Year, Infl_de, 'b:', 'LineWidth', 1);
% p_uk = plot(Year, Infl_uk, 'c:', 'LineWidth', 1);
% p_fr = plot(Year, Infl_fr, 'y:', 'LineWidth', 1);
% p_ca = plot(Year, Infl_ca, 'r:', 'LineWidth', 1);
% p_it = plot(Year, Infl_it, 'g:', 'LineWidth', 1);
% p_jp = plot(Year, Infl_jp, 'm:', 'LineWidth', 1);
% p_au = plot(Year, Infl_au, 'LineStyle', ':', 'Color', [1 0.5 0], 'LineWidth', 1);
% 
% xlim([1870 2024])
% xticks(1880:20:2020)
% ylim([-3 15])
% yticks(0:5:15)
% 
% legend([p_us p_de p_uk p_fr p_ca p_it p_jp p_au], ...
%     {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp', 'au'}, 'Location', 'southoutside',...
%     'Orientation', 'horizontal', 'Box', 'off')
% % title('$\bar{\pi}^w_t$ and $\pi_{it}$')
% 
% box on
% 
% printpdf(gcf, [figpath 'fig4a-Model1_Pibar-obs.pdf'])
%% Figure 4a: Trends and Observables for Inflation, Baseline Model
figure;
PlotStatesShaded(Year, qPi_bar); 
hold on;

codes = {'us','de','uk','fr','ca','it','jp','au','be','fi','ie','nl','no','ch','se','es','pt'};
labels = upper(codes);

country_colors = [
    0    0    0;
    0    0    1;
    0    1    1;
    1    1    0;
    1    0    0;
    0    1    0;
    1    0    1;
    1    0.5  0;
    0.5  0    0.5;
    0    0.5  0.5;
    0.5  0.5  0;
    0    0    0.5;
    0.7  0.3  0;
    0.3  0.3  0.3;
    1    0.7  0;
    0.8  0    0.2;
    0    0.7  0.3;
];

ph = gobjects(1, numel(codes));

for k = 1:numel(codes)
    c = codes{k};
    
    infl_data = eval(sprintf('Infl_%s', c));
    
    ph(k) = plot(Year, infl_data, ':', ...
        'Color', country_colors(k,:), ...
        'LineWidth', 1);
    
    eval(sprintf('p_%s = ph(k);', c));
end

% xlim([1870 2017])
% ylim([-3 15])
% yticks(0:5:15)

xlim([Year(1) Year(end)])
ylim([-3 6])
yticks(-2:2:6)

lgd = legend(ph, labels, ...
    'Interpreter', 'latex', ...
    'Location', 'southoutside', ...
    'FontSize', 9, ...
    'Orientation', 'horizontal', ...
    'NumColumns', 6);

legend boxoff
box on

printpdf(gcf, [figpath 'fig4a-Model1_Pibar-obs.pdf'])

%% Figure 4b: Trends and Observables for Inflation, Baseline Model
figure;
median = quantile(CommonTrends(:,2,:), .5, 3);
plotMedian = plot(Year, median, 'k--', 'LineWidth', 2, ...
    'HandleVisibility', 'off');
hold on; 
box on;

codes = {'us','de','uk','fr','ca','it','jp','au','be','fi','ie','nl','no','ch','se','es','pt'};
labels = upper(codes);

country_colors = [
    0    0    0;
    0    0    1;
    0    1    1;
    1    1    0;
    1    0    0;
    0    1    0;
    1    0    1;
    1    0.5  0;
    0.5  0    0.5;
    0    0.5  0.5;
    0.5  0.5  0;
    0    0    0.5;
    0.7  0.3  0;
    0.3  0.3  0.3;
    1    0.7  0;
    0.8  0    0.2;
    0    0.7  0.3;
];

ph = gobjects(1, numel(codes));

for k = 1:numel(codes)
    c = codes{k};
    
    var_name = sprintf('qPi_bar_%s', c);
    if exist(var_name, 'var')
        qPi_data = eval(sprintf('%s(:, 3)', var_name));
        
        ph(k) = plot(Year, qPi_data, ':', ...
            'Color', country_colors(k,:), ...
            'LineWidth', 1.5);
        
        eval(sprintf('p_%s = ph(k);', c));
    else
        fprintf('Warning: Variable %s not found\n', var_name);
    end
end

valid_handles = ph(~arrayfun(@(x) isa(x,'matlab.graphics.GraphicsPlaceholder'), ph));
valid_labels = labels(~arrayfun(@(x) isa(x,'matlab.graphics.GraphicsPlaceholder'), ph));

hline = refline(0);
hline.HandleVisibility = 'off';
hline.Color = 'k';

xlim([Year(1) Year(end)])
ylim([-3 20])
xticks(1880:20:2020)
yticks(0:5:20)

lgd = legend(ph, labels, ...
    'Interpreter', 'latex', ...
    'Location', 'southoutside', ...
    'FontSize', 9, ...
    'Orientation', 'horizontal', ...
    'NumColumns', 6);

legend boxon
box on

printpdf(gcf, [figpath 'fig4b-Model1_Pibar-countries.pdf'])


%% Figure 5: MY regression fit

fSize = 15;

MY = xlsread('../indata/Data_MY.xlsx');  % Load MY for each country
MY_us = MY(:,2);
MY_de = MY(:,3);
MY_uk = MY(:,4);
MY_fr = MY(:,5);
MY_ca = MY(:,6);
MY_it = MY(:,7);
MY_jp = MY(:,8);
MY_G7 = MY(:,9);

% Compute fitted values
fit_us_MY = [ones(T, 1) MY_us] * regress(qRshort_bar_us(:, 3), [ones(T, 1) MY_us]);
fit_de_MY = [ones(T, 1) MY_de] * regress(qRshort_bar_de(:, 3), [ones(T, 1) MY_de]);
fit_uk_MY = [ones(T, 1) MY_uk] * regress(qRshort_bar_uk(:, 3), [ones(T, 1) MY_uk]);
fit_fr_MY = [ones(T, 1) MY_fr] * regress(qRshort_bar_fr(:, 3), [ones(T, 1) MY_fr]);
fit_ca_MY = [ones(T, 1) MY_ca] * regress(qRshort_bar_ca(:, 3), [ones(T, 1) MY_ca]);
fit_it_MY = [ones(T, 1) MY_it] * regress(qRshort_bar_it(:, 3), [ones(T, 1) MY_it]);
fit_jp_MY = [ones(T, 1) MY_jp] * regress(qRshort_bar_jp(:, 3), [ones(T, 1) MY_jp]);
fit_G7_MY = [ones(T, 1) MY_G7] * regress(qRshort_bar   (:, 3), [ones(T, 1) MY_G7]);

figure
p_wd = PlotStatesShaded(Year, qRshort_bar); hold on;
plot(Year, fit_G7_MY, 'k-')
%title('World')
axis([Year(1) Year(end) -3 6])
set(gca, 'FontSize', fSize)
xticks(1880:20:2020)
printpdf(gcf, [figpath 'fig5-Model1_Rbar-countries_MY-fitted-common.pdf'])

figure
p_us = PlotStatesShaded(Year, qRshort_bar_us); hold on;
plot(Year, fit_us_MY, 'k-')
%title('US')
axis([Year(1) Year(end) -3 6])
set(gca, 'FontSize', fSize)
xticks(1880:20:2020)
printpdf(gcf, [figpath 'fig5-Model1_Rbar-countries_MY-fitted-us.pdf'])

figure
p_de = PlotStatesShaded(Year, qRshort_bar_de); hold on;
p_de.Color = 'b';
plot(Year, fit_de_MY, 'k-')
%title('Germany')
axis([Year(1) Year(end) -3 6])
set(gca, 'FontSize', fSize)
xticks(1880:20:2020)
printpdf(gcf, [figpath 'fig5-Model1_Rbar-countries_MY-fitted-de.pdf'])

figure
p_uk = PlotStatesShaded(Year, qRshort_bar_uk); hold on;
p_uk.Color = 'c';
plot(Year, fit_uk_MY, 'k-')
%title('U.K.')
axis([Year(1) Year(end) -3 6])
set(gca, 'FontSize', fSize)
xticks(1880:20:2020)
printpdf(gcf, [figpath 'fig5-Model1_Rbar-countries_MY-fitted-uk.pdf'])

figure
p_fr = PlotStatesShaded(Year, qRshort_bar_fr); hold on;
p_fr.Color = 'y';
plot(Year, fit_fr_MY, 'k-')
%title('France')
axis([Year(1) Year(end) -3 6])
set(gca, 'FontSize', fSize)
xticks(1880:20:2020)
printpdf(gcf, [figpath 'fig5-Model1_Rbar-countries_MY-fitted-fr.pdf'])

figure
p_ca = PlotStatesShaded(Year, qRshort_bar_ca); hold on;
p_ca.Color = 'r';
plot(Year, fit_ca_MY, 'k-')
%title('Canada')
axis([Year(1) Year(end) -3 6])
set(gca, 'FontSize', fSize)
xticks(1880:20:2020)
printpdf(gcf, [figpath 'fig5-Model1_Rbar-countries_MY-fitted-ca.pdf'])

figure
p_it = PlotStatesShaded(Year, qRshort_bar_it); hold on;
p_it.Color = 'g';
plot(Year, fit_it_MY, 'k-')
%title('Italy')
axis([Year(1) Year(end) -3 6])
set(gca, 'FontSize', fSize)
xticks(1880:20:2020)
printpdf(gcf, [figpath 'fig5-Model1_Rbar-countries_MY-fitted-it.pdf'])


figure
p_jp = PlotStatesShaded(Year, qRshort_bar_jp); hold on;
p_jp.Color = 'm';
plot(Year, fit_jp_MY, 'k-')
%title('Japan')
axis([Year(1) Year(end) -3 6])
set(gca, 'FontSize', fSize)
xticks(1880:20:2020)
printpdf(gcf, [figpath 'fig5-Model1_Rbar-countries_MY-fitted-jp.pdf'])




%% ---------------------- Appendix figures --------------------------------


%% A1: Trends and Observables for Term Spreads, Baseline Model

set(0, 'DefaultAxesFontSize',15)

f = figure;
filename = 'Tsbar';
h = PlotStatesShaded(Year, qTs_bar);
hold on; box on;
axis([Year(1) Year(end) -2 4]);

p_us = plot(Year, Ltir_us-Stir_us, 'k:', 'LineWidth', 1); hold on;
p_de = plot(Year, Ltir_de-Stir_de, 'b:', 'LineWidth', 1);
p_uk = plot(Year, Ltir_uk-Stir_uk, 'c:', 'LineWidth', 1);
p_fr = plot(Year, Ltir_fr-Stir_fr, 'y:', 'LineWidth', 1);
p_ca = plot(Year, Ltir_ca-Stir_ca, 'r:', 'LineWidth', 1);
p_it = plot(Year, Ltir_it-Stir_it, 'g:', 'LineWidth', 1);
p_jp = plot(Year, Ltir_jp-Stir_jp, 'm:', 'LineWidth', 1);
p_au = plot(Year, Ltir_au-Stir_au, 'LineStyle', ':', 'Color', [1 0.5 0], 'LineWidth', 1);

legend([p_us p_de p_uk p_fr p_ca p_it p_jp p_au],...
    {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp', 'au'},...
    'Interpreter','latex',...
    'Location','SouthOutside',...
    'FontSize',12,'Orientation', 'horizontal'); 

legend boxoff;

%title('$\overline{ts}^w_t$ and $R^L_{i,t} - R_{i,t}$', 'Interpreter', 'latex')

printpdf(gcf, [appenpath 'figa1a-Model1_Tsbar.pdf'])


%% A1: Trends and Observables for Term Spreads, Baseline Model


f = figure;
filename = 'Tsbar-countries';
h = PlotStatesShaded(Year, qTs_bar(:,3));
hold on; box on;
axis([Year(1) Year(end) -2 4]);
p_us = plot(Year, qTs_bar_us(:,3), 'k:', 'LineWidth', 2); hold on;
p_de = plot(Year, qTs_bar_de(:,3), 'b:', 'LineWidth', 2);
p_uk = plot(Year, qTs_bar_uk(:,3), 'c:', 'LineWidth', 2);
p_fr = plot(Year, qTs_bar_fr(:,3), 'y:', 'LineWidth', 2);
p_ca = plot(Year, qTs_bar_ca(:,3), 'r:', 'LineWidth', 2);
p_it = plot(Year, qTs_bar_it(:,3), 'g:', 'LineWidth', 2);
p_jp = plot(Year, qTs_bar_jp(:,3), 'm:', 'LineWidth', 2);
 p_au = plot(Year, qTs_bar_au(:,3), 'LineStyle', ':', 'Color', [1 0.5 0], 'LineWidth', 2);

legend([p_us p_de p_uk p_fr p_ca p_it p_jp p_au],...
    {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp', 'au'},...
    'Interpreter','latex',...
    'Location','SouthOutside',...
    'FontSize',12,'Orientation', 'horizontal'); 

legend boxoff;
%title('$\overline{ts}^w_t$ and $\overline{ts}_{i,t}$', 'Interpreter', 'latex')
printpdf(gcf, [appenpath 'figa1b-Model1_Tsbar-countries.pdf'])


%% A2: Country-Specfic Trends r_it and Observables, Baseline Model

fSize = 15;  % Font size

f = figure;

Rshort_country_average = ...
    mean([Stir_us-Infl_us, Stir_de-Infl_de, Stir_uk-Infl_uk,...
          Stir_fr-Infl_fr, Stir_ca-Infl_ca, Stir_it-Infl_it, ...
          Stir_jp-Infl_jp, Stir_au-Infl_au], 2, 'omitnan');
      
h = PlotStatesShaded(Year, qRshort_bar);
hold on; box on; axis([Year(1) Year(end) -3 6]);
%title('World', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-common.pdf'])

f = figure;
h = PlotStatesShaded(Year, qRshort_bar_us_idio); hold on;
p_us = plot(Year, qRshort_bar_us_idio(:,3), 'k:', 'LineWidth', 2);
plot(Year, Stir_us-Infl_us-Rshort_country_average, 'k:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('U.S.', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-us.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_de_idio); hold on;
h.Color = 'b';
h.LineStyle = '--';
p_de = plot(Year, qRshort_bar_de_idio(:,3), 'b:', 'LineWidth', 2);
plot(Year, Stir_de-Infl_de-Rshort_country_average, 'b:', 'LineWidth', 1);
 axis([Year(1) Year(end) -3 6]);
%title('Germany', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-de.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_uk_idio); hold on;
h.Color = 'c';
p_uk = plot(Year, qRshort_bar_uk_idio(:,3), 'c:', 'LineWidth', 2);
plot(Year, Stir_uk-Infl_uk-Rshort_country_average, 'c:', 'LineWidth', 1);
 axis([Year(1) Year(end) -3 6]);
%title('U.K.', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-uk.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_fr_idio); hold on;
h.Color = 'y';
p_fr = plot(Year, qRshort_bar_fr_idio(:,3), 'y:', 'LineWidth', 2);
plot(Year, Stir_fr-Infl_fr-Rshort_country_average, 'y:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('France', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-fr.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_ca_idio); hold on;
h.Color = 'r';
p_ca = plot(Year, qRshort_bar_ca_idio(:,3), 'r:', 'LineWidth', 2);
plot(Year, Stir_ca-Infl_ca-Rshort_country_average, 'r:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('Canada', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-ca.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_it_idio); hold on;
h.Color = 'g';
p_it = plot(Year, qRshort_bar_it_idio(:,3), 'g:', 'LineWidth', 2);
plot(Year, Stir_it-Infl_it-Rshort_country_average, 'g:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('Italy', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-it.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_jp_idio); hold on;
h.Color = 'm';
p_jp = plot(Year, qRshort_bar_jp_idio(:,3), 'm:', 'LineWidth', 2);
plot(Year, Stir_jp-Infl_jp-Rshort_country_average, 'm:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('Japan', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-jp.pdf'])


figure;
h = PlotStatesShaded(Year, qRshort_bar_au_idio); hold on;
h.Color = [1 0.5 0];
p_au = plot(Year, qRshort_bar_au_idio(:,3), 'LineStyle', ':', 'Color', [1 0.5 0], 'LineWidth', 2);
plot(Year, Stir_au-Infl_au-Rshort_country_average, 'LineStyle', ':', 'Color', [1 0.5 0], 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('Australia', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa2-Model1_Rshort-countries_trend-idio_obs-average-au.pdf'])
