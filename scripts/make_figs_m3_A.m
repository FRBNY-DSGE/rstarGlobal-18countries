cd /data/dsge_data_dir/rstarGlobal-18countries/scripts
try
    MainModel3_A_MakeFigures
    fprintf('FIGURES_DONE model 3 A\n');
catch ME
    fprintf(2, 'FIGURE SCRIPT ERROR (model 3 A): %s\n', getReport(ME));
end
exit
