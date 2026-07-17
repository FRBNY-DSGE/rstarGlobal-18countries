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

% Raw change draw-vectors {complabel, dGlobal_p1, dGlobal_p2, dUS_p1, dUS_p2}
% for the slide/star-format tables + replica generated at the end of the script.
comp1 = { '$\overline{r}_t$', ...
    Rshort_bar(t_end1,:)-Rshort_bar(t_start1,:), Rshort_bar(t_end2,:)-Rshort_bar(t_start2,:), ...
    Rshort_bar_us(t_end1,:)-Rshort_bar_us(t_start1,:), Rshort_bar_us(t_end2,:)-Rshort_bar_us(t_start2,:) };

% End-of-sample LEVEL of global and US r-bar at year t_end2 (the value plotted
% at the right edge of Figure 1), for the levels table below: median, 90%
% interval [5,95] and {P(r-bar<0)}, same fmtCI format as the change tables.
levels_m1 = [{'Global $\overline{r}^{w}_{t}$'; 'US $\overline{r}^{w}_{t}$'}, ...
    {fmtCI(Rshort_bar(t_end2,:), Quant); fmtCI(Rshort_bar_us(t_end2,:), Quant)}];

clearvars -except rows_m1 levels_m1 comp1

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

% Raw change draw-vectors for the slide/star tables (US "other m" reuses the
% global m-bar, mirroring the current-format table's US m row).
comp2 = { ...
    '$\overline{r}_t$', ...
        Rshort_bar(t_end1,:)-Rshort_bar(t_start1,:), Rshort_bar(t_end2,:)-Rshort_bar(t_start2,:), ...
        Rshort_bar_us(t_end1,:)-Rshort_bar_us(t_start1,:), Rshort_bar_us(t_end2,:)-Rshort_bar_us(t_start2,:); ...
    '$-\overline{cy}_t$', ...
        -(Cy_bar(t_end1,:)-Cy_bar(t_start1,:)), -(Cy_bar(t_end2,:)-Cy_bar(t_start2,:)), ...
        -(Cy_bar_us(t_end1,:)-Cy_bar_us(t_start1,:)), -(Cy_bar_us(t_end2,:)-Cy_bar_us(t_start2,:)); ...
    '$\overline{m}_t$', ...
        M_bar(t_end1,:)-M_bar(t_start1,:), M_bar(t_end2,:)-M_bar(t_start2,:), ...
        M_bar(t_end1,:)-M_bar(t_start1,:), M_bar(t_end2,:)-M_bar(t_start2,:) };

% End-of-sample LEVEL of global and US r-bar for Model 2 (same format as levels_m1).
levels_m2 = [{'Global $\overline{r}^{w}_{t}$'; 'US $\overline{r}^{w}_{t}$'}, ...
    {fmtCI(Rshort_bar(t_end2,:), Quant); fmtCI(Rshort_bar_us(t_end2,:), Quant)}];

clearvars -except rows_m1 levels_m1 rows_m2 levels_m2 comp1 comp2

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

% Raw change draw-vectors for the slide/star tables (US g and beta reuse the
% global g-bar/beta-bar, mirroring the current-format table's US g/beta rows).
comp3 = { ...
    '$\overline{r}_t$', ...
        Rshort_bar(t_end1,:)-Rshort_bar(t_start1,:), Rshort_bar(t_end2,:)-Rshort_bar(t_start2,:), ...
        Rshort_bar_us(t_end1,:)-Rshort_bar_us(t_start1,:), Rshort_bar_us(t_end2,:)-Rshort_bar_us(t_start2,:); ...
    '$\overline{g}_t$', ...
        G_bar(t_end1,:)-G_bar(t_start1,:), G_bar(t_end2,:)-G_bar(t_start2,:), ...
        G_bar(t_end1,:)-G_bar(t_start1,:), G_bar(t_end2,:)-G_bar(t_start2,:); ...
    '$\overline{\beta}_t$', ...
        Beta_bar(t_end1,:)-Beta_bar(t_start1,:), Beta_bar(t_end2,:)-Beta_bar(t_start2,:), ...
        Beta_bar(t_end1,:)-Beta_bar(t_start1,:), Beta_bar(t_end2,:)-Beta_bar(t_start2,:); ...
    '$-\overline{cy}_t$', ...
        -(Cy_bar(t_end1,:)-Cy_bar(t_start1,:)), -(Cy_bar(t_end2,:)-Cy_bar(t_start2,:)), ...
        -(Cy_bar_us(t_end1,:)-Cy_bar_us(t_start1,:)), -(Cy_bar_us(t_end2,:)-Cy_bar_us(t_start2,:)) };

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

%% Table: Global and US r* -- Model 3 ALTERNATIVE version A (original inflation prior 2)
% Same construction as the default Model 3 block above but reading
% OutputModel3_A.mat (the alternative whose inflation trend prior keeps the
% original code's value 2). Emitted only if that estimate exists. Writes
% GlobalUS_Model3_A.tex (standalone).
if exist('../results/OutputModel3_A.mat', 'file')
    clear States Trends AA QQ RR CC CommonTrends   % free the default Model 3 (B) arrays before loading A
    load('../results/OutputModel3_A.mat');
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
    fid = fopen('../tables/GlobalUS_Model3_A.tex', 'w');
    WriteTeXTable(fid, {'', '1990-2019', '2019-2025'}, 'l|c|c', [row_labels x], ...
        'GlobalUS Model3 A (alternative: original inflation prior 2)\\ \\');
    fclose(fid);
end

%% Slide/star-format tables (both layouts, both intervals)
% These MAP the current tables via Table 1's footnote rule in the Rachel
% discussion: each cell becomes median^{stars} (interval), where the stars come
% from the posterior probability the change is in the expected direction --
% P(change<0) for 1990-2019, P(change>0) for 2019-2025 -- exceeding 0.90/0.95/
% 0.975 (*/**/***). Two interval widths are emitted: 90% (5/95, matches the
% deck) and 95% (2.5/97.5, matches the footnote text).
%   Layout A: same rows/cols as the current tables, appended into each
%             GlobalUS_Model{1,2,3}.tex.
%   Layout B: exact deck replica (r^w and r^US column-groups, model panels) in
%             GlobalUS_SlideReplica.tex.
append_star_tables('../tables/GlobalUS_Model1.tex', 'GlobalUS Model1', comp1);
append_star_tables('../tables/GlobalUS_Model2.tex', 'GlobalUS Model2', comp2);
append_star_tables('../tables/GlobalUS_Model3.tex', 'GlobalUS Model3', comp3);

fid = fopen('../tables/GlobalUS_SlideReplica.tex', 'w');
for cc = {[0.050 0.500 0.950], '90'; [0.025 0.500 0.975], '95'}'
    fprintf(fid, '\\begin{table}[htpb!]\n');
    fprintf(fid, 'Global and US $\\overline{r}^{w}_{t}$ -- slide format, %s\\%% CI\\\\ \\\\\n\\centering\n', cc{2});
    write_replica(fid, comp1, comp2, comp3, cc{1});
    fprintf(fid, '\\end{table}\n\n');
end
fclose(fid);


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

function s = fmtStars(d, Quant, dir)
% fmtStars  Slide-format cell: median^{stars} (lo,hi). Stars from the posterior
% probability the change is in the expected direction exceeding 0.90/0.95/0.975
% (*/**/***): dir='below' -> P(d<0) (1990-2019 decline); dir='above' -> P(d>0)
% (2019-2025 rise). This is exactly the rule in Table 1's footnote (Rachel disc.).
M  = numel(d);
z  = sort(d); z = z(ceil(Quant * M));
if strcmp(dir, 'below'); p = sum(d < 0) / M; else; p = sum(d > 0) / M; end
st = ''; if p > 0.975; st = '***'; elseif p > 0.95; st = '**'; elseif p > 0.90; st = '*'; end
s = ['\makecell{' sprintf('$ %0.2f^{%s} $ \\\\ $ (%0.2f, %0.2f) $', z(2), st, z(1), z(3)) '}'];
end

function append_star_tables(fname, disp_name, comp)
% Append the slide-format table (Layout A: same rows/cols as the current table,
% Global rows then US rows) to fname, at 90% and 95% CI.
% comp: N x 5 cell {complabel, dGlobal_p1, dGlobal_p2, dUS_p1, dUS_p2}.
header = {'', '1990-2019', '2019-2025'}; style = 'l|c|c';
n = size(comp, 1);
labels = [cellfun(@(c) ['Global ' c], comp(:,1), 'UniformOutput', false); ...
          cellfun(@(c) ['US ' c],     comp(:,1), 'UniformOutput', false)];
fid = fopen(fname, 'a');
for cc = {[0.050 0.500 0.950], '90'; [0.025 0.500 0.975], '95'}'
    Q = cc{1}; xs = cell(2*n, 2);
    for i = 1:n
        xs{i,1}   = fmtStars(comp{i,2}, Q, 'below'); xs{i,2}   = fmtStars(comp{i,3}, Q, 'above');
        xs{n+i,1} = fmtStars(comp{i,4}, Q, 'below'); xs{n+i,2} = fmtStars(comp{i,5}, Q, 'above');
    end
    WriteTeXTable(fid, header, style, [labels xs], [disp_name ' (slide format, ' cc{2} '\% CI)\\ \\']);
end
fclose(fid);
end

function write_replica(fid, comp1, comp2, comp3, Q)
% Deck-style replica (Layout B): r^w and r^US as column groups, model panels as
% rows, slide-format star cells. Written directly (not via WriteTeXTable).
fprintf(fid, '\\begin{tabular}{l cc cc}\n');
fprintf(fid, ' & \\multicolumn{2}{c}{$\\overline{r}^{w}_{t}$} & \\multicolumn{2}{c}{$\\overline{r}^{US}_{t}$}\\\\\n');
fprintf(fid, ' & 1990--2019 & 2019--2025 & 1990--2019 & 2019--2025\\\\\n\\hline\\hline\n');
panels = {'Baseline Model', comp1; 'Convenience Yield Model', comp2; 'Consumption Model', comp3};
for pp = 1:size(panels, 1)
    fprintf(fid, '\\multicolumn{5}{@{}l}{\\textit{%s}}\\\\\n', panels{pp, 1});
    comp = panels{pp, 2};
    for i = 1:size(comp, 1)
        fprintf(fid, '%s & %s & %s & %s & %s\\\\\n', comp{i, 1}, ...
            fmtStars(comp{i, 2}, Q, 'below'), fmtStars(comp{i, 3}, Q, 'above'), ...
            fmtStars(comp{i, 4}, Q, 'below'), fmtStars(comp{i, 5}, Q, 'above'));
    end
end
fprintf(fid, '\\hline\n\\end{tabular}\n');
end
