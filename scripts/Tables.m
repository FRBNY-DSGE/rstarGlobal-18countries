%% Tables.m    Global and US r* summary tables (Model 1 and Model 2)
% Companion to makeTables.m. Reports the change in global and US r-bar over
% 1990-2019 and 2019-2025: posterior median, 90% interval (5th/95th pct, as
% in the Liberty Street blog table), and P(change<0) per cell (rather than
% makeTables.m's full multi-band stat block).
% For Model 2, r-bar is additionally decomposed into r*, -cy, and "other"
% (m-bar), at both the global and US level.
clear;
addpath('Routines')

% NOTE: Quant must be (re)defined AFTER each load() below -- the Output*.mat
% files contain their own saved 5-element `Quant` variable from the
% estimation script, which silently overwrites any value set before load.

%% Table: Global and US r* -- Model 1 (Baseline)
% Same construction as makeTables.m Table A1a (US r*), plus the global r*
% trend it's built from, just median + 95% CI instead of the full
% multi-band stat block.

model      = 'Model1';
model_name = 'GlobalUS_Model1';

load(['../results/18/Output' model '.mat']);

% 90% equal-tailed interval (5th/95th pct) -- matches the Liberty Street
% blog table "Pre- and Post-COVID Changes in R*". Must come after load().
Quant = [0.050 0.500 0.950];

M = size(CommonTrends, 3); %#ok<*NASGU>

Rshort_bar         = squeeze(CommonTrends(:, 1, :));
Rshort_bar_us_idio = squeeze(CommonTrends(:, 4, :));
Rshort_bar_us      = repmat(transpose(squeeze(CC(1, 1, :))), T, 1) .* Rshort_bar + Rshort_bar_us_idio;

t_start1 = find(Year == 1990);
t_end1   = find(Year == 2019);
t_start2 = find(Year == 2019);
t_end2   = find(Year == 2025);

x = {}; %#ok<*AGROW>

y = {};
y = [y; {fmtCI(Rshort_bar(t_end1,:)    - Rshort_bar(t_start1,:),    Quant)}];
y = [y; {fmtCI(Rshort_bar_us(t_end1,:) - Rshort_bar_us(t_start1,:), Quant)}];
x = [x y];

y = {};
y = [y; {fmtCI(Rshort_bar(t_end2,:)    - Rshort_bar(t_start2,:),    Quant)}];
y = [y; {fmtCI(Rshort_bar_us(t_end2,:) - Rshort_bar_us(t_start2,:), Quant)}];
x = [x y];

row_labels = {'Global $\overline{r}^{w}_{t}$'; 'US $\overline{r}^{w}_{t}$'};
style      = 'l|c|c';
header     = {'', '1990-2019', '2019-2025'};

fid = fopen(['../tables/' model_name '.tex'], 'w');
WriteTeXTable(fid, header, style, [row_labels x], [strrep(model_name, '_', ' ') '\\ \\']);
fclose(fid);

rows_m1 = [row_labels x];  % keep Model 1 rows for the combined table below

% End-of-sample LEVEL of global and US r-bar at year t_end2 (the value plotted
% at the right edge of Figure 1), for the levels table below: median, 90%
% interval [5,95] and {P(r-bar<0)}, same fmtCI format as the change tables.
levels_m1 = [{'Global $\overline{r}^{w}_{t}$'; 'US $\overline{r}^{w}_{t}$'}, ...
    {fmtCI(Rshort_bar(t_end2,:), Quant); fmtCI(Rshort_bar_us(t_end2,:), Quant)}];

clearvars -except rows_m1 levels_m1

%% Table: Global and US r* -- Model 2 (Convenience Yield), decomposed into r*, -cy, other

model      = 'Model2';
model_name = 'GlobalUS_Model2';

load(['../results/18/Output' model '.mat']);

% Redefine after load -- the .mat contains its own `Quant` (see note above).
Quant = [0.050 0.500 0.950];

M = size(CommonTrends, 3);

M_bar      = squeeze(CommonTrends(:, 1, :));
Cy_bar     = squeeze(CommonTrends(:, 4, :));
Rshort_bar = M_bar - Cy_bar;

Rshort_bar_us_idio = squeeze(CommonTrends(:, 5, :));
Rshort_bar_us      = repmat(transpose(squeeze(CC(1, 1, :))), T, 1) .* Rshort_bar + Rshort_bar_us_idio;
% Defined so that Rshort_bar_us = M_bar - Cy_bar_us, mirroring the global
% identity Rshort_bar = M_bar - Cy_bar (there is no US-specific M_bar, so
% makeTables.m's US table reuses the *global* M_bar for the "other" row,
% and we do the same here).
Cy_bar_us          = Cy_bar - (Rshort_bar_us - Rshort_bar);

t_start1 = find(Year == 1990);
t_end1   = find(Year == 2019);
t_start2 = find(Year == 2019);
t_end2   = find(Year == 2025);

x = {};

y = {};
y = [y; {fmtCI(Rshort_bar(t_end1,:)    - Rshort_bar(t_start1,:),    Quant)}];
y = [y; {fmtCI(M_bar(t_end1,:)         - M_bar(t_start1,:),         Quant)}];
y = [y; {fmtCI(-(Cy_bar(t_end1,:)      - Cy_bar(t_start1,:)),       Quant)}];
y = [y; {fmtCI(Rshort_bar_us(t_end1,:) - Rshort_bar_us(t_start1,:), Quant)}];
y = [y; {fmtCI(M_bar(t_end1,:)         - M_bar(t_start1,:),         Quant)}];
y = [y; {fmtCI(-(Cy_bar_us(t_end1,:)   - Cy_bar_us(t_start1,:)),    Quant)}];
x = [x y];

y = {};
y = [y; {fmtCI(Rshort_bar(t_end2,:)    - Rshort_bar(t_start2,:),    Quant)}];
y = [y; {fmtCI(M_bar(t_end2,:)         - M_bar(t_start2,:),         Quant)}];
y = [y; {fmtCI(-(Cy_bar(t_end2,:)      - Cy_bar(t_start2,:)),       Quant)}];
y = [y; {fmtCI(Rshort_bar_us(t_end2,:) - Rshort_bar_us(t_start2,:), Quant)}];
y = [y; {fmtCI(M_bar(t_end2,:)         - M_bar(t_start2,:),         Quant)}];
y = [y; {fmtCI(-(Cy_bar_us(t_end2,:)   - Cy_bar_us(t_start2,:)),    Quant)}];
x = [x y];

row_labels = {'Global $\overline{r}^{w}_{t}$'; 'Global other ($\overline{m}^{w}_{t}$)'; ...
    'Global $-\overline{cy}^{w}_{t}$'; 'US $\overline{r}^{w}_{t}$'; ...
    'US other ($\overline{m}^{w}_{t}$)'; 'US $-\overline{cy}^{w}_{t}$'};
style      = 'l|c|c';
header     = {'', '1990-2019', '2019-2025'};

fid = fopen(['../tables/' model_name '.tex'], 'w');
WriteTeXTable(fid, header, style, [row_labels x], [strrep(model_name, '_', ' ') '\\ \\']);
fclose(fid);

rows_m2 = [row_labels x];  % keep Model 2 rows for the combined table below

% End-of-sample LEVEL of global and US r-bar for Model 2 (same format as levels_m1).
levels_m2 = [{'Global $\overline{r}^{w}_{t}$'; 'US $\overline{r}^{w}_{t}$'}, ...
    {fmtCI(Rshort_bar(t_end2,:), Quant); fmtCI(Rshort_bar_us(t_end2,:), Quant)}];

clearvars -except rows_m1 levels_m1 rows_m2 levels_m2

%% Table: Global and US r* -- Model 3 (Consumption), decomposed into g, beta, -cy
% World identity: r-bar = g-bar + beta-bar - cy-bar. The US rows reuse the world
% g-bar and beta-bar with a US-specific cy-bar that absorbs the US idiosyncratic
% real-rate trend, mirroring the Model 2 US construction above. Model 3's output
% lives in results/ (not results/18) as OutputModel3_new.mat.

model_name = 'GlobalUS_Model3';

load('../results/OutputModel3_new.mat');

Quant = [0.050 0.500 0.950];   % redefine after load (the .mat carries its own Quant)

M = size(CommonTrends, 3);

G_bar      = squeeze(CommonTrends(:, 1, :));
Cy_bar     = squeeze(CommonTrends(:, 4, :));
Beta_bar   = squeeze(CommonTrends(:, 5, :));
Rshort_bar = G_bar + Beta_bar - Cy_bar;

Rshort_bar_us_idio = squeeze(CommonTrends(:, 7, :));  % 6 world trends, so US rs_idio is col 7
Rshort_bar_us      = repmat(transpose(squeeze(CC(1, 1, :))), T, 1) .* Rshort_bar + Rshort_bar_us_idio;
Cy_bar_us          = Cy_bar - (Rshort_bar_us - Rshort_bar);

t_start1 = find(Year == 1990);
t_end1   = find(Year == 2019);
t_start2 = find(Year == 2019);
t_end2   = find(Year == 2025);

x = {};

y = {};
y = [y; {fmtCI(Rshort_bar(t_end1,:)    - Rshort_bar(t_start1,:),    Quant)}];
y = [y; {fmtCI(G_bar(t_end1,:)         - G_bar(t_start1,:),         Quant)}];
y = [y; {fmtCI(Beta_bar(t_end1,:)      - Beta_bar(t_start1,:),      Quant)}];
y = [y; {fmtCI(-(Cy_bar(t_end1,:)      - Cy_bar(t_start1,:)),       Quant)}];
y = [y; {fmtCI(Rshort_bar_us(t_end1,:) - Rshort_bar_us(t_start1,:), Quant)}];
y = [y; {fmtCI(G_bar(t_end1,:)         - G_bar(t_start1,:),         Quant)}];
y = [y; {fmtCI(Beta_bar(t_end1,:)      - Beta_bar(t_start1,:),      Quant)}];
y = [y; {fmtCI(-(Cy_bar_us(t_end1,:)   - Cy_bar_us(t_start1,:)),    Quant)}];
x = [x y];

y = {};
y = [y; {fmtCI(Rshort_bar(t_end2,:)    - Rshort_bar(t_start2,:),    Quant)}];
y = [y; {fmtCI(G_bar(t_end2,:)         - G_bar(t_start2,:),         Quant)}];
y = [y; {fmtCI(Beta_bar(t_end2,:)      - Beta_bar(t_start2,:),      Quant)}];
y = [y; {fmtCI(-(Cy_bar(t_end2,:)      - Cy_bar(t_start2,:)),       Quant)}];
y = [y; {fmtCI(Rshort_bar_us(t_end2,:) - Rshort_bar_us(t_start2,:), Quant)}];
y = [y; {fmtCI(G_bar(t_end2,:)         - G_bar(t_start2,:),         Quant)}];
y = [y; {fmtCI(Beta_bar(t_end2,:)      - Beta_bar(t_start2,:),      Quant)}];
y = [y; {fmtCI(-(Cy_bar_us(t_end2,:)   - Cy_bar_us(t_start2,:)),    Quant)}];
x = [x y];

row_labels = {'Global $\overline{r}^{w}_{t}$'; 'Global $\overline{g}^{w}_{t}$'; ...
    'Global $\overline{\beta}^{w}_{t}$'; 'Global $-\overline{cy}^{w}_{t}$'; ...
    'US $\overline{r}^{w}_{t}$'; 'US $\overline{g}^{w}_{t}$'; ...
    'US $\overline{\beta}^{w}_{t}$'; 'US $-\overline{cy}^{w}_{t}$'};
style      = 'l|c|c';
header     = {'', '1990-2019', '2019-2025'};

fid = fopen(['../tables/' model_name '.tex'], 'w');
WriteTeXTable(fid, header, style, [row_labels x], [strrep(model_name, '_', ' ') '\\ \\']);
fclose(fid);

rows_m3 = [row_labels x];

%% Table: Combined Model 1 + Model 2 + Model 3 (all panels in one table)
% Panel labels span all 3 columns via WriteTeXTable's NaN multicolumn rule.

body = [{'\textbf{Model 1 (Baseline)}',          NaN, NaN}; ...
        rows_m1; ...
        {'\textbf{Model 2 (Convenience yield)}', NaN, NaN}; ...
        rows_m2; ...
        {'\textbf{Model 3 (Consumption)}',       NaN, NaN}; ...
        rows_m3];

fid = fopen('../tables/GlobalUS_Combined.tex', 'w');
WriteTeXTable(fid, header, style, body, ...
    'Global and US $\overline{r}^{w}_{t}$: Models 1, 2 and 3\\ \\');
fclose(fid);

%% Table: End-of-sample LEVELS of global and US r-bar (both models)
% The value plotted at the last year of Figure 1 (Model 1) and its Model 2
% counterpart -- median, 90% interval [5,95], {P(r-bar<0)}.
yl = num2str(Year(t_end2));
body_lvl = [{'\textbf{Model 1 (Baseline)}',          NaN}; ...
            levels_m1; ...
            {'\textbf{Model 2 (Convenience yield)}', NaN}; ...
            levels_m2];

fid = fopen('../tables/GlobalUS_Levels.tex', 'w');
WriteTeXTable(fid, {'', yl}, 'l|c', body_lvl, ...
    ['Global and US $\overline{r}^{w}_{t}$ level in ' yl '\\ \\']);
fclose(fid);

%% Table: Global and US r* -- Model 3 VERSION B (Baa fix + inflation prior sqrt(2))
% Same construction as the Model 3 block above but reading OutputModel3_B.mat
% (the variant whose inflation trend prior is aligned to Models 1 & 2). Emitted
% only if that estimate exists. Writes GlobalUS_Model3_B.tex (standalone).
if exist('../results/OutputModel3_B.mat', 'file')
    clear States Trends AA QQ RR CC CommonTrends   % free the Model 3 (A) arrays before loading B
    load('../results/OutputModel3_B.mat');
    Quant = [0.050 0.500 0.950];
    M = size(CommonTrends, 3);
    G_bar      = squeeze(CommonTrends(:, 1, :));
    Cy_bar     = squeeze(CommonTrends(:, 4, :));
    Beta_bar   = squeeze(CommonTrends(:, 5, :));
    Rshort_bar = G_bar + Beta_bar - Cy_bar;
    Rshort_bar_us_idio = squeeze(CommonTrends(:, 7, :));
    Rshort_bar_us      = repmat(transpose(squeeze(CC(1, 1, :))), T, 1) .* Rshort_bar + Rshort_bar_us_idio;
    Cy_bar_us          = Cy_bar - (Rshort_bar_us - Rshort_bar);
    t_start1 = find(Year == 1990); t_end1 = find(Year == 2019);
    t_start2 = find(Year == 2019); t_end2 = find(Year == 2025);
    x = {};
    y = {};
    y = [y; {fmtCI(Rshort_bar(t_end1,:)    - Rshort_bar(t_start1,:),    Quant)}];
    y = [y; {fmtCI(G_bar(t_end1,:)         - G_bar(t_start1,:),         Quant)}];
    y = [y; {fmtCI(Beta_bar(t_end1,:)      - Beta_bar(t_start1,:),      Quant)}];
    y = [y; {fmtCI(-(Cy_bar(t_end1,:)      - Cy_bar(t_start1,:)),       Quant)}];
    y = [y; {fmtCI(Rshort_bar_us(t_end1,:) - Rshort_bar_us(t_start1,:), Quant)}];
    y = [y; {fmtCI(G_bar(t_end1,:)         - G_bar(t_start1,:),         Quant)}];
    y = [y; {fmtCI(Beta_bar(t_end1,:)      - Beta_bar(t_start1,:),      Quant)}];
    y = [y; {fmtCI(-(Cy_bar_us(t_end1,:)   - Cy_bar_us(t_start1,:)),    Quant)}];
    x = [x y];
    y = {};
    y = [y; {fmtCI(Rshort_bar(t_end2,:)    - Rshort_bar(t_start2,:),    Quant)}];
    y = [y; {fmtCI(G_bar(t_end2,:)         - G_bar(t_start2,:),         Quant)}];
    y = [y; {fmtCI(Beta_bar(t_end2,:)      - Beta_bar(t_start2,:),      Quant)}];
    y = [y; {fmtCI(-(Cy_bar(t_end2,:)      - Cy_bar(t_start2,:)),       Quant)}];
    y = [y; {fmtCI(Rshort_bar_us(t_end2,:) - Rshort_bar_us(t_start2,:), Quant)}];
    y = [y; {fmtCI(G_bar(t_end2,:)         - G_bar(t_start2,:),         Quant)}];
    y = [y; {fmtCI(Beta_bar(t_end2,:)      - Beta_bar(t_start2,:),      Quant)}];
    y = [y; {fmtCI(-(Cy_bar_us(t_end2,:)   - Cy_bar_us(t_start2,:)),    Quant)}];
    x = [x y];
    row_labels = {'Global $\overline{r}^{w}_{t}$'; 'Global $\overline{g}^{w}_{t}$'; ...
        'Global $\overline{\beta}^{w}_{t}$'; 'Global $-\overline{cy}^{w}_{t}$'; ...
        'US $\overline{r}^{w}_{t}$'; 'US $\overline{g}^{w}_{t}$'; ...
        'US $\overline{\beta}^{w}_{t}$'; 'US $-\overline{cy}^{w}_{t}$'};
    fid = fopen('../tables/GlobalUS_Model3_B.tex', 'w');
    WriteTeXTable(fid, {'', '1990-2019', '2019-2025'}, 'l|c|c', [row_labels x], ...
        'GlobalUS Model3 B (inflation prior aligned to Models 1 \& 2)\\ \\');
    fclose(fid);
end


function s = fmtCI(x, Quant)
% fmtCI  Format a draw vector as a \makecell with the posterior median,
% the coverage interval implied by Quant([1 end]), and the posterior
% probability that the change is below zero, {P(dx<0)}. The blog's
% significance stars are P(dx<0) for 1990-2019 declines and
% 1 - P(dx<0) for 2019-2024 increases (>0.90/0.95/0.975 = */**/***).
M   = numel(x);
iQ  = ceil(Quant * M);
z   = sort(x);
z   = z(iQ);
p   = sum(x < 0) / M;
s = ['\makecell{' sprintf('$ %0.2f $ \\\\ $ [%0.2f, %0.2f] $ \\\\ $ \\{%0.3f\\} $', z(2), z(1), z(3), p) '}'];
end
