% Model 2
% Generate figure 6, A13-A17

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

codes = {'us','de','uk','fr','ca','it','jp','au','be','fi','ie','nl','no','ch','se','es','pt', 'dk'};
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


%% Figure 6a: r-bar^w

f = figure;

h= bands(Year, qRshort_bar);
box on
xticks(1880:20:2020)
axis([Year(1) Year(end) -3 6])
%title('$\overline{r}^w_t$', 'Interpreter', 'latex')

printpdf(gcf, [figpath 'fig6a-Model2_Rbar.pdf'])


%% Figure 6b: r-bar^w and -cy-bar^w_t

figure
tmax = find(Year==1870);
% -qCy_bar is normalized
h = bands(Year, qRshort_bar, -qCy_bar + (qCy_bar(tmax,3) + qRshort_bar(tmax,3)));
hold on
xticks(1880:20:2020)
axis([Year(1) Year(end) -3 6])  % Changed from -3 3 to -2 5
yticks(-3:1:6)  % Changed from -3:0.5:3 to -2:1:5 for cleaner spacing
%title('$\overline{r}^w_t$ and $-\overline{cy}_t^w$', 'Interpreter', 'latex')
box on
printpdf(gcf, [figpath 'fig6b-Model2_Rbar-Cybar-allcountries.pdf'])

%% Figure 6c: r-bar^w_t and m-bar^w_t

figure
tmax=155
% -qCy_bar is normalized
h = bands(Year, qRshort_bar, qM_bar - (qM_bar(tmax, 3) - qRshort_bar(tmax, 3)));

axis([Year(1) Year(end) -3 6])
yticks(-3:1:6)
%title('$\overline{r}^w_t$ and $\overline{m}_t^w$', 'Interpreter', 'latex')
box on
xticks(1880:20:2020)

printpdf(gcf, [figpath 'fig6c-Model2_Rbar-Mbar-allcountries.pdf'])


%% Table Components: r-ber^w and its components
% Define analysis periods
periods = [
    1980, 2019;
    2019, 2024
];

% Define variables to analyze (matching the table structure)
variables = struct();
variables(1).name = 'Delta_rw';
variables(1).index = 1;  % Rshort_bar (world real rate)
variables(1).description = 'r̄ʷ';
variables(1).transform = @(x) x;  % No transformation

variables(2).name = 'Delta_cyw';
variables(2).index = 4;  % Cy_bar (convenience yield)
variables(2).description = '-c̄yʷ';
variables(2).transform = @(x) -x;  % Negative transformation

variables(3).name = 'Delta_mw';
variables(3).index = 1;  % M_bar (will be calculated as Rshort_bar + Cy_bar)
variables(3).description = 'm̄ʷ';
variables(3).transform = @(x, cy) x + cy;  % m̄ʷ = r̄ʷ + c̄yʷ

% Initialize results structure
Table1 = struct();
counter = 1;

% Loop over variables and periods
for v = 1:length(variables)
    for p = 1:size(periods, 1)
        start_year = periods(p, 1);
        end_year = periods(p, 2);
        
        % Calculate change over the period (all draws)
        start_idx = find(Year == start_year, 1);
        end_idx = find(Year == end_year, 1);
        
        if isempty(start_idx) || isempty(end_idx)
            warning('Year %d or %d not found in data', start_year, end_year);
            continue;
        end
        
        % Extract data based on variable type
        if v == 3  % m̄ʷ = r̄ʷ + c̄yʷ
            start_rw = squeeze(CommonTrends(start_idx, 1, :));  % Rshort_bar
            end_rw = squeeze(CommonTrends(end_idx, 1, :));
            start_cy = squeeze(CommonTrends(start_idx, 4, :));  % Cy_bar  
            end_cy = squeeze(CommonTrends(end_idx, 4, :));
            
            start_data = start_rw + start_cy;
            end_data = end_rw + end_cy;
        else
            % Regular variables
            start_data = squeeze(CommonTrends(start_idx, variables(v).index, :));
            end_data = squeeze(CommonTrends(end_idx, variables(v).index, :));
            
            % Apply transformation
            if v == 2  % -c̄yʷ
                start_data = variables(v).transform(start_data);
                end_data = variables(v).transform(end_data);
            end
        end
        
        % Calculate change
        change_data = end_data - start_data;
        
        % Store results
        Table1(counter).(variables(v).name) = change_data;
        Table1(counter).Years = [start_year, end_year];
        Table1(counter).Variable = variables(v).description;
        Table1(counter).VarIndex = v;
        
        counter = counter + 1;
    end
end

% Display results and create formatted table
clc

% Initialize storage for table data
table_data = {};
row_labels = {};
var_labels = {};

% Process each result
result_counter = 1;
for v = 1:length(variables)
    for p = 1:size(periods, 1)
        if result_counter > length(Table1)
            break;
        end
        
        period_str = sprintf('%d-%d', Table1(result_counter).Years(1), Table1(result_counter).Years(2));
        var_name = variables(v).name;
        data = Table1(result_counter).(var_name);
        
        % Calculate statistics
        median_val = quantile(data, 0.5);
        ci_lower = quantile(data, 0.05);
        ci_upper = quantile(data, 0.95);
        
        % Calculate p-value (two-tailed test that change = 0)
        prob_positive = mean(data > 0);
        prob_negative = mean(data < 0);
        p_value = 2 * min(prob_positive, prob_negative);
        
        % Store data for table
        table_data{result_counter, 1} = sprintf('%.2f', median_val);
        table_data{result_counter, 2} = sprintf('(%.2f, %.2f)', ci_lower, ci_upper);
        table_data{result_counter, 3} = sprintf('%.3f', p_value);
        
        if p == 1  % First period for this variable
            var_labels{result_counter} = Table1(result_counter).Variable;
        else
            var_labels{result_counter} = '';  % Empty for subsequent periods
        end
        row_labels{result_counter} = period_str;
        
        % Also display in console
        fprintf('\n-----------------[%s: %s]------------------\n', ...
            Table1(result_counter).Variable, period_str);
        fprintf('Median: %.4f\n', median_val);
        fprintf('90%% posterior coverage: (%.4f, %.4f)\n', ci_lower, ci_upper);
        fprintf('P-value (H0: change = 0): %.4f\n', p_value);
        
        result_counter = result_counter + 1;
    end
end

% Create and display formatted table
fprintf('\n\n=== FORMATTED TABLE ===\n');
fprintf('%-8s | %-12s | %-10s | %-20s | %-10s\n', 'Variable', 'Period', 'Median', '90% CI', 'P-value');
fprintf('%s\n', repmat('-', 1, 68));
for j = 1:length(Table1)
    fprintf('%-8s | %-12s | %-10s | %-20s | %-10s\n', ...
        var_labels{j}, row_labels{j}, table_data{j, 1}, table_data{j, 2}, table_data{j, 3});
end

% Generate LaTeX table
latex_filename = 'world_trends_changes.tex';
fid = fopen(latex_filename, 'w');

fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Changes in World Economic Trends}\n');
fprintf(fid, '\\label{tab:world_trends_changes}\n');
fprintf(fid, '\\begin{tabular}{llccc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'Variable & Period & Median & 90\\%% Posterior Interval & P-value \\\\\n');
fprintf(fid, '\\midrule\n');

for j = 1:length(Table1)
    if ~isempty(var_labels{j})  % Print variable name only for first occurrence
        fprintf(fid, '%s & %s & %s & %s & %s \\\\\n', ...
            var_labels{j}, row_labels{j}, table_data{j, 1}, table_data{j, 2}, table_data{j, 3});
    else  % Empty variable column for subsequent periods
        fprintf(fid, ' & %s & %s & %s & %s \\\\\n', ...
            row_labels{j}, table_data{j, 1}, table_data{j, 2}, table_data{j, 3});
    end
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\begin{tablenotes}\n');
fprintf(fid, '\\small\n');
fprintf(fid, '\\item Notes: Changes in world economic trends computed from posterior draws.\n');
fprintf(fid, '\\item 90\\%% posterior intervals show the 5th and 95th percentiles.\n');
fprintf(fid, '\\item P-values test the null hypothesis that the change equals zero (two-tailed).\n');
fprintf(fid, '\\item $\\overline{r}^w$: world real interest rate; $-\\overline{cy}^w$: negative world convenience yield; $\\overline{m}^w$: world nominal rate.\n');
fprintf(fid, '\\end{tablenotes}\n');
fprintf(fid, '\\end{table}\n');

fclose(fid);

fprintf('\nLaTeX table saved to: %s\n', latex_filename);


%% -------------------------- APPENDIX FIGURES ---------------------------- %%



%% A13: Trends in Global and U.S. Real Rates


figure

PlotStatesShaded(Year, qRshort_bar)
hold on
plot(Year, qRshort_bar_us(:,3), ...
    'k:', 'LineWidth', 1.5);  % r-bar US

axis([Year(1) Year(end) -3 6])
%title('$\overline{r}^w_t$ and $\overline{r}_{US, t}$', 'Interpreter', 'latex')
box on

printpdf(gcf, [appenpath 'figa13-Model2_Rshortbar-us.pdf'])

%% A14a: Trends and Observables for Short-Term Real Rates, Convenience Yield Model


figure

PlotStatesShaded(Year, qRshort_bar);
hold on;
p_us = plot(Year, Stir_us - Infl_us, 'k:', 'LineWidth', 1);
p_de = plot(Year, Stir_de - Infl_de, 'b:', 'LineWidth', 1);
p_uk = plot(Year, Stir_uk - Infl_uk, 'c:', 'LineWidth', 1);
p_fr = plot(Year, Stir_fr - Infl_fr, 'y:', 'LineWidth', 1);
p_ca = plot(Year, Stir_ca - Infl_ca, 'r:', 'LineWidth', 1);
p_it = plot(Year, Stir_it - Infl_it, 'g:', 'LineWidth', 1);
p_jp = plot(Year, Stir_jp - Infl_jp, 'm:', 'LineWidth', 1);

axis([Year(1) Year(end) -6 12])
yticks(-5:5:10)

legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
    {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'},...
    'Interpreter', 'latex',...
    'Location',    'SouthOutside',...
    'FontSize',    12,...
    'Orientation', 'horizontal');
legend boxoff;
%title('$\overline{r}^w_t$ and $R_{i,t} - \pi_{i,t}$', 'Interpreter', 'latex')

printpdf(gcf, [appenpath 'figa14a-Model2_Rshortbar-obs.pdf'])

%% A14b: Trends and Observables for Short-Term Real Rates, Convenience Yield Model

figure
hold on

p_us = plot(Year, qRshort_bar_us(:,3), 'k:', 'LineWidth', 2); hold on;
p_de = plot(Year, qRshort_bar_de(:,3), 'b:', 'LineWidth', 2);
p_uk = plot(Year, qRshort_bar_uk(:,3), 'c:', 'LineWidth', 2);
p_fr = plot(Year, qRshort_bar_fr(:,3), 'y:', 'LineWidth', 2);
p_ca = plot(Year, qRshort_bar_ca(:,3), 'r:', 'LineWidth', 2);
p_it = plot(Year, qRshort_bar_it(:,3), 'g:', 'LineWidth', 2);
p_jp = plot(Year, qRshort_bar_jp(:,3), 'm:', 'LineWidth', 2);
plot(Year, qRshort_bar(:, 3), 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');


hline                  = refline(0);
hline.HandleVisibility = 'off';
hline.Color            = 'k';

xlim([Year(137) Year(end)])
xticks(2008:4:2024)
ylim([-3 3])
yticks(-2:0.5:2)

legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
    {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'},...
    'Interpreter', 'latex',...
    'Location',    'SouthOutside',...
    'FontSize',    12,...
    'Orientation', 'horizontal');

%title('$\overline{r}^w_t$ and $\overline{r}_{i,t}$', 'Interpreter','latex')

legend boxoff;
box on;

printpdf(gcf, [appenpath 'figa14b-Model2_Rshortbar-countries-recent.pdf'])

%% A15a: Trends and Observables for Inflation, Convenience Yield Model


figure;

PlotStatesShaded(Year, qPi_bar); hold on;  % pi-bar world
p_us = plot(Year, Infl_us, 'k:', 'LineWidth', 1); hold on;
p_de = plot(Year, Infl_de, 'b:', 'LineWidth', 1);
p_uk = plot(Year, Infl_uk, 'c:', 'LineWidth', 1);
p_fr = plot(Year, Infl_fr, 'y:', 'LineWidth', 1);
p_ca = plot(Year, Infl_ca, 'r:', 'LineWidth', 1);
p_it = plot(Year, Infl_it, 'g:', 'LineWidth', 1);
p_jp = plot(Year, Infl_jp, 'm:', 'LineWidth', 1);

xlim([1870 2017])
ylim([-3 15])
yticks(0:5:15)

legend([p_us p_de p_uk p_fr p_ca p_it p_jp], ...
    {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'}, 'Location', 'southoutside',...
    'Orientation', 'horizontal', 'Box', 'off')
%title('$\overline{\pi}^w_t$ and $\pi_{it}$')

box on

printpdf(gcf, [appenpath 'figa15a-Model2_pibar-obs.pdf'])

%% A15b: Trends and Observables for Inflation, Convenience Yield Model

figure;

median = quantile(CommonTrends(:,2,:), .5, 3);

plotMedian = plot(Year, median, 'k--', 'LineWidth', 2, ...
    'HandleVisibility', 'off');
hold on; box on;

p_us = plot(Year, qPi_bar_us(:, 3), 'k:', 'LineWidth', 1.5); hold on;
p_de = plot(Year, qPi_bar_de(:, 3), 'b:', 'LineWidth', 1.5);
p_uk = plot(Year, qPi_bar_uk(:, 3), 'c:', 'LineWidth', 1.5);
p_fr = plot(Year, qPi_bar_fr(:, 3), 'y:', 'LineWidth', 1.5);
p_ca = plot(Year, qPi_bar_ca(:, 3), 'r:', 'LineWidth', 1.5);
p_it = plot(Year, qPi_bar_it(:, 3), 'g:', 'LineWidth', 1.5);
p_jp = plot(Year, qPi_bar_jp(:, 3), 'm:', 'LineWidth', 1.5);


hline = refline(0);
hline.HandleVisibility = 'off';
hline.Color = 'k';
axis([Year(1) Year(end) -3 15]);

%title('$\overline{\pi}^w_t$ and $\overline{\pi}_{i,t}$', 'Interpreter', 'latex')

legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
    {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'}, 'Location', 'southoutside',...
    'Orientation', 'horizontal', 'Box', 'off')
box on
yticks(0:5:15)

printpdf(gcf, [appenpath 'figa15b-Model2_pibar-countries.pdf'])

%% A16a: Trends and Observables for Term Spreads, Convenience Yield Model

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

legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
    {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'},...
    'Interpreter','latex',...
    'Location','SouthOutside',...
    'FontSize',12,'Orientation', 'horizontal'); 

legend boxoff;

%title('$\overline{ts}^w_t$ and $R^L_{i,t} - R_{i,t}$', 'Interpreter', 'latex')

printpdf(gcf, [appenpath 'figa16a-Model2_Rbar-Tsbar.pdf'])

%% A16b: Trends and Observables for Term Spreads, Convenience Yield Model

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

legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
    {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'},...
    'Interpreter','latex',...
    'Location','SouthOutside',...
    'FontSize',12,'Orientation', 'horizontal'); 

legend boxoff;
%title('$\overline{ts}^w_t$ and $\overline{ts}_{i,t}$', 'Interpreter', 'latex')

printpdf(gcf, [appenpath 'figa16b-Model2_Rbar-tsbar-countries.pdf'])

% A17: Country-Specfic Trends r_it and Observables, Convenience Yield Model
fSize = 15;  % Font size

f = figure;

Rshort_country_average = ...
    mean([Stir_us-Infl_us, Stir_de-Infl_de, Stir_uk-Infl_uk,...
          Stir_fr-Infl_fr, Stir_ca-Infl_ca, Stir_it-Infl_it, ...
          Stir_jp-Infl_jp], 2, 'omitnan');

h = PlotStatesShaded(Year, qRshort_bar);
hold on; box on; axis([Year(1) Year(end) -3 6]);
%title('World', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa17-Model2_Rshort-countries_trend-idio_obs-average-common.pdf'])

f = figure;
h = PlotStatesShaded(Year, qRshort_bar_us_idio); hold on;
p_us = plot(Year, qRshort_bar_us_idio(:,3), 'k:', 'LineWidth', 2);
plot(Year, Stir_us-Infl_us-Rshort_country_average, 'k:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('U.S.', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa17-Model2_Rshort-countries_trend-idio_obs-average-us.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_de_idio); hold on;
h.Color = 'b';
h.LineStyle = '--';
p_de = plot(Year, qRshort_bar_de_idio(:,3), 'b:', 'LineWidth', 2);
plot(Year, Stir_de-Infl_de-Rshort_country_average, 'b:', 'LineWidth', 1);
 axis([Year(1) Year(end) -3 6]);
%title('Germany', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa17-Model2_Rshort-countries_trend-idio_obs-average-de.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_uk_idio); hold on;
h.Color = 'c';
p_uk = plot(Year, qRshort_bar_uk_idio(:,3), 'c:', 'LineWidth', 2);
plot(Year, Stir_uk-Infl_uk-Rshort_country_average, 'c:', 'LineWidth', 1);
 axis([Year(1) Year(end) -3 6]);
%title('U.K.', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa17-Model2_Rshort-countries_trend-idio_obs-average-uk.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_fr_idio); hold on;
h.Color = 'y';
p_fr = plot(Year, qRshort_bar_fr_idio(:,3), 'y:', 'LineWidth', 2);
plot(Year, Stir_fr-Infl_fr-Rshort_country_average, 'y:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('France', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa17-Model2_Rshort-countries_trend-idio_obs-average-fr.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_ca_idio); hold on;
h.Color = 'r';
p_ca = plot(Year, qRshort_bar_ca_idio(:,3), 'r:', 'LineWidth', 2);
plot(Year, Stir_ca-Infl_ca-Rshort_country_average, 'r:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('Canada', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa17-Model2_Rshort-countries_trend-idio_obs-average-ca.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_it_idio); hold on;
h.Color = 'g';
p_it = plot(Year, qRshort_bar_it_idio(:,3), 'g:', 'LineWidth', 2);
plot(Year, Stir_it-Infl_it-Rshort_country_average, 'g:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('Italy', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa17-Model2_Rshort-countries_trend-idio_obs-average-it.pdf'])

figure;
h = PlotStatesShaded(Year, qRshort_bar_jp_idio); hold on;
h.Color = 'm';
p_jp = plot(Year, qRshort_bar_jp_idio(:,3), 'm:', 'LineWidth', 2);
plot(Year, Stir_jp-Infl_jp-Rshort_country_average, 'm:', 'LineWidth', 1);
axis([Year(1) Year(end) -3 6]);
%title('Japan', 'Interpreter', 'latex')
set(gca, 'FontSize', fSize)
printpdf(gcf, [appenpath 'figa17-Model2_Rshort-countries_trend-idio_obs-average-jp.pdf'])

