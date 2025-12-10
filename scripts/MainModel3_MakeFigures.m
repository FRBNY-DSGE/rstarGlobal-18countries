% Model 3   Creates figures for consumption model (7, A18-A22)

%% Preliminaries

clc;
close all

addpath('Routines')
load('../results/OutputModel3_new.mat')

figpath = '../figures/';
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
tmax = find(Year==1870);

% trends

%% Extract Common Trends

codes = {'us','de','uk','fr','ca','it','jp','au','be','fi','ie','nl','no','ch','se','es','pt'};
Nc = numel(codes);
T = size(CommonTrends, 1);
Nd = size(CommonTrends, 3);

G_bar = squeeze(CommonTrends(:, 1, :));
Pi_bar = squeeze(CommonTrends(:, 2, :));
Ts_bar = squeeze(CommonTrends(:, 3, :));
Cy_bar = squeeze(CommonTrends(:, 4, :));
Beta_bar = squeeze(CommonTrends(:, 5, :));
Gamma_bar = squeeze(CommonTrends(:, 6, :));
M_bar = G_bar + Beta_bar;
Rshort_bar = M_bar - Cy_bar;
Rlong_bar = Rshort_bar + Ts_bar;
DC_bar = G_bar + Gamma_bar;

%% Extract Country-Specific Idiosyncratic Components

base = 6;
idx_rs = base + (1:Nc);
idx_pi = base + Nc + (1:Nc);
idx_ts = base + 2*Nc + (1:Nc);
idx_dc = base + 3*Nc + (1:Nc);

for i = 1:Nc
    code = codes{i};
    
    Rs_idio_i = squeeze(CommonTrends(:, idx_rs(i), :));
    Pi_idio_i = squeeze(CommonTrends(:, idx_pi(i), :));
    Ts_idio_i = squeeze(CommonTrends(:, idx_ts(i), :));
    DC_idio_i = squeeze(CommonTrends(:, idx_dc(i), :));
    
    eval(sprintf('Rshort_bar_%s_idio = Rs_idio_i;', code));
    eval(sprintf('Pi_bar_%s_idio = Pi_idio_i;', code));
    eval(sprintf('Ts_bar_%s_idio = Ts_idio_i;', code));
    eval(sprintf('DC_bar_%s_idio = DC_idio_i;', code));
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
    DC_idio_i = eval(sprintf('DC_bar_%s_idio', code));
    
    Rshort_i = w_rs .* Rshort_bar + Rs_idio_i;
    Pi_i = w_pi .* Pi_bar + Pi_idio_i;
    Ts_i = Ts_bar + Ts_idio_i;
    Rlong_i = Rlong_bar + Rs_idio_i + Ts_idio_i;
    DC_i = DC_bar + DC_idio_i;
    
    eval(sprintf('Rshort_bar_%s = Rshort_i;', code));
    eval(sprintf('Pi_bar_%s = Pi_i;', code));
    eval(sprintf('Ts_bar_%s = Ts_i;', code));
    eval(sprintf('Rlong_bar_%s = Rlong_i;', code));
    eval(sprintf('DC_bar_%s = DC_i;', code));
end

% sorted trends

%% Sort Common Trends

sRshort_bar = sort(Rshort_bar, 2);
sPi_bar = sort(Pi_bar, 2);
sTs_bar = sort(Ts_bar, 2);
sRlong_bar = sort(Rlong_bar, 2);
sCy_bar = sort(Cy_bar, 2);
sG_bar = sort(G_bar, 2);
sBeta_bar = sort(Beta_bar, 2);
sGamma_bar = sort(Gamma_bar, 2);
sDC_bar = sort(DC_bar, 2);

%% Sort Country-Specific Trends

for k = 1:Nc
    c = codes{k};
    eval(sprintf('sRshort_bar_%s = sort(Rshort_bar_%s, 2);', c, c));
    eval(sprintf('sRlong_bar_%s = sort(Rlong_bar_%s, 2);', c, c));
    eval(sprintf('sPi_bar_%s = sort(Pi_bar_%s, 2);', c, c));
    eval(sprintf('sTs_bar_%s = sort(Ts_bar_%s, 2);', c, c));
    eval(sprintf('sDC_bar_%s = sort(DC_bar_%s, 2);', c, c));
    eval(sprintf('sRshort_bar_%s_idio = sort(Rshort_bar_%s_idio, 2);', c, c));
end

%% Calculate Quantiles of Common Trends

M = size(sRshort_bar, 2);
qInd = max(1, min(M, ceil(Quant(:)' .* M)));

qRshort_bar = sRshort_bar(:, qInd);
qPi_bar = sPi_bar(:, qInd);
qTs_bar = sTs_bar(:, qInd);
qRlong_bar = sRlong_bar(:, qInd);
qCy_bar = sCy_bar(:, qInd);
qG_bar = sG_bar(:, qInd);
qBeta_bar = sBeta_bar(:, qInd);
qGamma_bar = sGamma_bar(:, qInd);
qDC_bar = sDC_bar(:, qInd);

%% Calculate Quantiles of Country-Specific Trends

for k = 1:numel(codes)
    c = codes{k};
    eval(sprintf('qRshort_bar_%s = sRshort_bar_%s(:, qInd);', c, c));
    eval(sprintf('qRlong_bar_%s = sRlong_bar_%s(:, qInd);', c, c));
    eval(sprintf('qPi_bar_%s = sPi_bar_%s(:, qInd);', c, c));
    eval(sprintf('qTs_bar_%s = sTs_bar_%s(:, qInd);', c, c));
    eval(sprintf('qDC_bar_%s = sDC_bar_%s(:, qInd);', c, c));
    eval(sprintf('qRshort_bar_%s_idio = sRshort_bar_%s_idio(:, qInd);', c, c));
end


%% Figure 7a

f = figure; 
filename = 'Rbar-Cybar';
h = bands(Year, qRshort_bar, -qCy_bar- (-qCy_bar(tmax,3) - qRshort_bar(tmax,3)));
box on; 
axis([Year(1) Year(end) -3.0 +6.0]); 
xticks(1880:20:2020);

%title('$\overline{r}^w_t$ and $-\overline{cy}_t^w$', 'Interpreter', 'latex')

printpdf(gcf, [figpath 'fig7a-Model3_Rbar-Cybar.pdf'])


%% Figure 7b

f = figure; 
filename = 'Rbar-Gbar';
h = bands(Year, qRshort_bar, qG_bar - (qG_bar(tmax,3)-qRshort_bar(tmax,3)));
box on; 
axis([Year(1) Year(end) -3.0 +6.0]); 
xticks(1880:20:2020)
%title('$\overline{r}^w_t$ and $\overline{g}_t^w$', 'Interpreter', 'latex')

printpdf(gcf, [figpath 'fig7b-Model3_Rbar-Gbar.pdf'])


%% Figure 7c

f = figure; 
filename = 'Rbar-Betabar';
h = bands(Year, qRshort_bar, qBeta_bar - (qBeta_bar(tmax,3) - qRshort_bar(tmax,3)));
box on; 
axis([Year(1) Year(end) -3.0 6.0]);
xticks(1880:20:2020)
%title('$\overline{r}^w_t$ and $\overline{\beta}_t^w$', 'Interpreter', 'latex')

printpdf(gcf, [figpath 'fig7c-Model3_Rbar-Betabar.pdf'])


%% -------------------------- APPENDIX FIGURES -----------------------------

%% A18: Trends in Global and U.S. Real Rates: 1870-2016, Consumption Model
% 
% figure
% 
% PlotStatesShaded(Year, qRshort_bar)
% hold on
% plot(Year, qRshort_bar_us(:, 3), ...
%     'k:', 'LineWidth', 1.5);  % r-bar US
% 
% axis([Year(1) Year(end) -3 6])
% %title('$\bar{r}^w_t$ and $\bar{r}_{US, t}$', 'Interpreter', 'latex')
% box on
% 
% printpdf(gcf, [appenpath 'figa18-Model2_Rshortbar-us.pdf'])
% 
% 
% %% A19a: Trends and Observables for Short-Term Real Rates, Consumption Model
% 
% figure
% 
% PlotStatesShaded(Year, qRshort_bar);
% hold on;
% p_us = plot(Year, Stir_us - Infl_us, 'k:', 'LineWidth', 1);
% p_de = plot(Year, Stir_de - Infl_de, 'b:', 'LineWidth', 1);
% p_uk = plot(Year, Stir_uk - Infl_uk, 'c:', 'LineWidth', 1);
% p_fr = plot(Year, Stir_fr - Infl_fr, 'y:', 'LineWidth', 1);
% p_ca = plot(Year, Stir_ca - Infl_ca, 'r:', 'LineWidth', 1);
% p_it = plot(Year, Stir_it - Infl_it, 'g:', 'LineWidth', 1);
% p_jp = plot(Year, Stir_jp - Infl_jp, 'm:', 'LineWidth', 1);
% 
% axis([Year(1) Year(end) -6 12])
% yticks(-5:5:10)
% 
% legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
%     {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'},...
%     'Interpreter', 'latex',...
%     'Location',    'SouthOutside',...
%     'FontSize',    12,...
%     'Orientation', 'horizontal');
% legend boxoff;
% %title('$\overline{r}^w_t$ and $R_{i,t} - \pi_{i,t}$', 'Interpreter', 'latex')
% 
% printpdf(gcf, [appenpath 'figa19a-Model2_Rshortbar-obs.pdf'])
% 
% %% A19b: Trends and Observables for Short-Term Real Rates, Consumption Model
% 
% figure
% hold on
% 
% p_us = plot(Year, qRshort_bar_us(:,3), 'k:', 'LineWidth', 2); hold on;
% p_de = plot(Year, qRshort_bar_de(:,3), 'b:', 'LineWidth', 2);
% p_uk = plot(Year, qRshort_bar_uk(:,3), 'c:', 'LineWidth', 2);
% p_fr = plot(Year, qRshort_bar_fr(:,3), 'y:', 'LineWidth', 2);
% p_ca = plot(Year, qRshort_bar_ca(:,3), 'r:', 'LineWidth', 2);
% p_it = plot(Year, qRshort_bar_it(:,3), 'g:', 'LineWidth', 2);
% p_jp = plot(Year, qRshort_bar_jp(:,3), 'm:', 'LineWidth', 2);
% plot(Year, qRshort_bar(:, 3), 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
% 
% 
% hline                  = refline(0);
% hline.HandleVisibility = 'off';
% hline.Color            = 'k';
% 
% xlim([Year(1) Year(end)])
% ylim([-3 6])
% yticks(-2:2:6)
% 
% legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
%     {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'},...
%     'Interpreter', 'latex',...
%     'Location',    'SouthOutside',...
%     'FontSize',    12,...
%     'Orientation', 'horizontal');
% 
% %title('$\overline{r}^w_t$ and $\overline{r}_{i,t}$', 'Interpreter','latex')
% 
% legend boxoff;
% box on;
% 
% printpdf(gcf, [appenpath 'figa19b-Model2_Rshortbar-countries.pdf'])
% 
% %% A20a: Trends and Observables for Inflation, Consumption Model
% 
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
% 
% xlim([1870 2017])
% ylim([-3 15])
% yticks(0:5:15)
% 
% legend([p_us p_de p_uk p_fr p_ca p_it p_jp], ...
%     {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'}, 'Location', 'southoutside',...
%     'Orientation', 'horizontal', 'Box', 'off')
% %title('$\overline{\pi}^w_t$ and $\pi_{it}$')
% 
% box on
% 
% printpdf(gcf, [appenpath 'figa20a-Model3_pibar-obs.pdf'])
% 
% 
% %% A20b: Trends and Observables for Inflation, Consumption Model
% 
% 
% figure;
% 
% median = quantile(CommonTrends(:,2,:), .5, 3);
% 
% plotMedian = plot(Year, median, 'k--', 'LineWidth', 2, ...
%     'HandleVisibility', 'off');
% hold on; box on;
% 
% p_us = plot(Year, qPi_bar_us(:, 3), 'k:', 'LineWidth', 1.5); hold on;
% p_de = plot(Year, qPi_bar_de(:, 3), 'b:', 'LineWidth', 1.5);
% p_uk = plot(Year, qPi_bar_uk(:, 3), 'c:', 'LineWidth', 1.5);
% p_fr = plot(Year, qPi_bar_fr(:, 3), 'y:', 'LineWidth', 1.5);
% p_ca = plot(Year, qPi_bar_ca(:, 3), 'r:', 'LineWidth', 1.5);
% p_it = plot(Year, qPi_bar_it(:, 3), 'g:', 'LineWidth', 1.5);
% p_jp = plot(Year, qPi_bar_jp(:, 3), 'm:', 'LineWidth', 1.5);
% 
% 
% hline = refline(0);
% hline.HandleVisibility = 'off';
% hline.Color = 'k';
% axis([Year(1) Year(end) -3 15]);
% 
% %title('$\overline{\pi}^w_t$ and $\overline{\pi}_{i,t}$', 'Interpreter', 'latex')
% 
% legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
%     {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'}, 'Location', 'southoutside',...
%     'Orientation', 'horizontal', 'Box', 'off')
% box on
% yticks(0:5:15)
% 
% printpdf(gcf, [appenpath 'figa20b-Model3_pibar-countries.pdf'])
% 
% %% A21a: Trends and Observables for Term Spreads, Consumption Model
% 
% 
% f = figure;
% filename = 'Tsbar';
% h = PlotStatesShaded(Year, qTs_bar);
% hold on; box on;
% axis([Year(1) Year(end) -2 4]);
% 
% p_us = plot(Year, Ltir_us-Stir_us, 'k:', 'LineWidth', 1); hold on;
% p_de = plot(Year, Ltir_de-Stir_de, 'b:', 'LineWidth', 1);
% p_uk = plot(Year, Ltir_uk-Stir_uk, 'c:', 'LineWidth', 1);
% p_fr = plot(Year, Ltir_fr-Stir_fr, 'y:', 'LineWidth', 1);
% p_ca = plot(Year, Ltir_ca-Stir_ca, 'r:', 'LineWidth', 1);
% p_it = plot(Year, Ltir_it-Stir_it, 'g:', 'LineWidth', 1);
% p_jp = plot(Year, Ltir_jp-Stir_jp, 'm:', 'LineWidth', 1);
% 
% legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
%     {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'},...
%     'Interpreter','latex',...
%     'Location','SouthOutside',...
%     'FontSize',12,'Orientation', 'horizontal'); 
% 
% legend boxoff;
% 
% %title('$\overline{ts}^w_t$ and $R^L_{i,t} - R_{i,t}$', 'Interpreter', 'latex')
% 
% printpdf(gcf, [appenpath 'figa21a-Model3_Rbar-Tsbar.pdf'])
% 
% %% A21b: Trends and Observables for Term Spreads, Consumption Model
% 
% f = figure;
% filename = 'Tsbar-countries';
% h = PlotStatesShaded(Year, qTs_bar(:,3));
% hold on; box on;
% axis([Year(1) Year(end) -2 4]);
% p_us = plot(Year, qTs_bar_us(:,3), 'k:', 'LineWidth', 2); hold on;
% p_de = plot(Year, qTs_bar_de(:,3), 'b:', 'LineWidth', 2);
% p_uk = plot(Year, qTs_bar_uk(:,3), 'c:', 'LineWidth', 2);
% p_fr = plot(Year, qTs_bar_fr(:,3), 'y:', 'LineWidth', 2);
% p_ca = plot(Year, qTs_bar_ca(:,3), 'r:', 'LineWidth', 2);
% p_it = plot(Year, qTs_bar_it(:,3), 'g:', 'LineWidth', 2);
% p_jp = plot(Year, qTs_bar_jp(:,3), 'm:', 'LineWidth', 2);
% 
% legend([p_us p_de p_uk p_fr p_ca p_it p_jp],...
%     {'us', 'de', 'uk', 'fr', 'ca', 'it', 'jp'},...
%     'Interpreter','latex',...
%     'Location','SouthOutside',...
%     'FontSize',12,'Orientation', 'horizontal'); 
% 
% legend boxoff;
% %title('$\overline{ts}^w_t$ and $\overline{ts}_{i,t}$', 'Interpreter', 'latex')
% 
% printpdf(gcf, [appenpath 'figa21b-Model3_Rbar-tsbar-countries.pdf'])
% 
% %% A22: Country-Specfic Trends r_it and Observables, Consumption Model
% 
% fSize = 15;  % Font size
% 
% f = figure;
% 
% Rshort_country_average = ...
%     mean([Stir_us-Infl_us, Stir_de-Infl_de, Stir_uk-Infl_uk,...
%           Stir_fr-Infl_fr, Stir_ca-Infl_ca, Stir_it-Infl_it, ...
%           Stir_jp-Infl_jp], 2, 'omitnan');
% 
% h = PlotStatesShaded(Year, qRshort_bar);
% hold on; box on; axis([Year(1) Year(end) -3 6]);
% %title('World', 'Interpreter', 'latex')
% set(gca, 'FontSize', fSize)
% printpdf(gcf, [appenpath 'figa22-Model3_Rshort-countries_trend-idio_obs-average-common.pdf'])
% 
% f = figure;
% h = PlotStatesShaded(Year, qRshort_bar_us_idio); hold on;
% p_us = plot(Year, qRshort_bar_us_idio(:,3), 'k:', 'LineWidth', 2);
% plot(Year, Stir_us-Infl_us-Rshort_country_average, 'k:', 'LineWidth', 1);
% axis([Year(1) Year(end) -3 6]);
% %title('U.S.', 'Interpreter', 'latex')
% set(gca, 'FontSize', fSize)
% printpdf(gcf, [appenpath 'figa22-Model3_Rshort-countries_trend-idio_obs-average-us.pdf'])
% 
% figure;
% h = PlotStatesShaded(Year, qRshort_bar_de_idio); hold on;
% h.Color = 'b';
% h.LineStyle = '--';
% p_de = plot(Year, qRshort_bar_de_idio(:,3), 'b:', 'LineWidth', 2);
% plot(Year, Stir_de-Infl_de-Rshort_country_average, 'b:', 'LineWidth', 1);
%  axis([Year(1) Year(end) -3 6]);
% %title('Germany', 'Interpreter', 'latex')
% set(gca, 'FontSize', fSize)
% printpdf(gcf, [appenpath 'figa22-Model3_Rshort-countries_trend-idio_obs-average-de.pdf'])
% 
% figure;
% h = PlotStatesShaded(Year, qRshort_bar_uk_idio); hold on;
% h.Color = 'c';
% p_uk = plot(Year, qRshort_bar_uk_idio(:,3), 'c:', 'LineWidth', 2);
% plot(Year, Stir_uk-Infl_uk-Rshort_country_average, 'c:', 'LineWidth', 1);
%  axis([Year(1) Year(end) -3 6]);
% %title('U.K.', 'Interpreter', 'latex')
% set(gca, 'FontSize', fSize)
% printpdf(gcf, [appenpath 'figa22-Model3_Rshort-countries_trend-idio_obs-average-uk.pdf'])
% 
% figure;
% h = PlotStatesShaded(Year, qRshort_bar_fr_idio); hold on;
% h.Color = 'y';
% p_fr = plot(Year, qRshort_bar_fr_idio(:,3), 'y:', 'LineWidth', 2);
% plot(Year, Stir_fr-Infl_fr-Rshort_country_average, 'y:', 'LineWidth', 1);
% axis([Year(1) Year(end) -3 6]);
% %title('France', 'Interpreter', 'latex')
% set(gca, 'FontSize', fSize)
% printpdf(gcf, [appenpath 'figa22-Model3_Rshort-countries_trend-idio_obs-average-fr.pdf'])
% 
% figure;
% h = PlotStatesShaded(Year, qRshort_bar_ca_idio); hold on;
% h.Color = 'r';
% p_ca = plot(Year, qRshort_bar_ca_idio(:,3), 'r:', 'LineWidth', 2);
% plot(Year, Stir_ca-Infl_ca-Rshort_country_average, 'r:', 'LineWidth', 1);
% axis([Year(1) Year(end) -3 6]);
% %title('Canada', 'Interpreter', 'latex')
% set(gca, 'FontSize', fSize)
% printpdf(gcf, [appenpath 'figa22-Model3_Rshort-countries_trend-idio_obs-average-ca.pdf'])
% 
% figure;
% h = PlotStatesShaded(Year, qRshort_bar_it_idio); hold on;
% h.Color = 'g';
% p_it = plot(Year, qRshort_bar_it_idio(:,3), 'g:', 'LineWidth', 2);
% plot(Year, Stir_it-Infl_it-Rshort_country_average, 'g:', 'LineWidth', 1);
% axis([Year(1) Year(end) -3 6]);
% %title('Italy', 'Interpreter', 'latex')
% set(gca, 'FontSize', fSize)
% printpdf(gcf, [appenpath 'figa22-Model3_Rshort-countries_trend-idio_obs-average-it.pdf'])
% 
% figure;
% h = PlotStatesShaded(Year, qRshort_bar_jp_idio); hold on;
% h.Color = 'm';
% p_jp = plot(Year, qRshort_bar_jp_idio(:,3), 'm:', 'LineWidth', 2);
% plot(Year, Stir_jp-Infl_jp-Rshort_country_average, 'm:', 'LineWidth', 1);
% axis([Year(1) Year(end) -3 6]);
% %title('Japan', 'Interpreter', 'latex')
% set(gca, 'FontSize', fSize)
% printpdf(gcf, [appenpath 'figa22-Model3_Rshort-countries_trend-idio_obs-average-jp.pdf'])
