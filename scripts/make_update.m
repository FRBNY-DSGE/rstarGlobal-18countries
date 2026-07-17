% make_update.m -- regenerate the public update/ CSVs + PNGs from MODEL 1 (baseline):
% global r*, US r*, and 18 country-specific r* (all Model 1), plus the two README PNGs.
% 18-country file renamed qRshort_bar_m2 -> qRshort_bar_m1 (it is Model 1, not Model 2).
addpath('Routines');
Quant = [0.025 0.160 0.500 0.840 0.975];
codes = {'us','de','uk','fr','ca','it','jp','au','be','fi','ie','nl','no','ch','se','es','pt','dk'};
Nc = numel(codes);
load('../results/18/OutputModel1.mat','CommonTrends','CC','Year');
[T,~,M] = size(CommonTrends);
qI = max(1, min(M, ceil(Quant*M)));
Rg = squeeze(CommonTrends(:,1,:));                 % global r* (world rs_wrd)
qg = q5(Rg, qI);
base = 3;                                          % Model 1: 3 world trends
Q18 = []; med18 = zeros(T,Nc); qUS = [];
for i = 1:Nc
    Rs_idio = squeeze(CommonTrends(:, base+i, :));
    w = repmat(transpose(squeeze(CC(i,1,:))), T, 1);
    Ri = w .* Rg + Rs_idio;
    qi = q5(Ri, qI);
    Q18 = [Q18 qi]; med18(:,i) = qi(:,3);
    if i == 1, qUS = qi; end                        % US = country 1
end
hdr = 'Year,r* 2.5 percentile,r* 16 percentile,r* median,r* 84 percentile,r* 97.5 percentile';
wcsv('../update/qRshort_bar_global_m1.csv', hdr, Year, qg);
wcsv('../update/qRshort_bar_us_m1.csv',     hdr, Year, qUS);
wcsv18('../update/qRshort_bar_m1.csv', codes, Year, Q18);
% PNG 1: US/Global r* (Model 1, Figure-1 style: global bands+median, US median dotted)
f1 = figure('visible','off','Position',[100 100 700 450]);
PlotStatesShaded(Year, qg); hold on;
plot(Year, qUS(:,3), 'k:', 'LineWidth', 1.5);
axis([1880 max(Year) -3 6]); xticks(1880:20:2020); box on;
saveas(f1, '../update/qRshort_bar_us_global_m1.png');
% PNG 2: 18 country-specific r* medians
f2 = figure('visible','off','Position',[100 100 800 450]);
plot(Year, med18); hold on;
axis([1880 max(Year) -6 8]); xticks(1880:20:2020); box on;
legend(upper(codes), 'Location','eastoutside', 'FontSize', 7);
saveas(f2, '../update/qRshort_bar_m1.png');
fprintf('UPDATE_DONE: wrote 3 CSVs + 2 PNGs to update/ (Year %d-%d)\n', Year(1), Year(end));

function q = q5(X, qI); s = sort(X,2); q = s(:,qI); end
function wcsv(f, hdr, Year, Q)
    fid = fopen(f,'w'); fprintf(fid,'%s\n',hdr);
    for t = 1:numel(Year); fprintf(fid,'%d',Year(t)); fprintf(fid,',%.9g',Q(t,:)); fprintf(fid,'\n'); end
    fclose(fid);
end
function wcsv18(f, codes, Year, Q)
    Nc = numel(codes); ql = {'r* 2.5 percentile','r* 16 percentile','r* median','r* 84 percentile','r* 97.5 percentile'};
    fid = fopen(f,'w');
    fprintf(fid,'Year'); for i=1:Nc; for j=1:5; fprintf(fid,',%s',codes{i}); end; end; fprintf(fid,'\n');
    for i=1:Nc; for j=1:5; fprintf(fid,',%s',ql{j}); end; end; fprintf(fid,'\n');
    for t=1:numel(Year); fprintf(fid,'%d',Year(t)); fprintf(fid,',%.15g',Q(t,:)); fprintf(fid,'\n'); end
    fclose(fid);
end
