cd /data/dsge_data_dir/rstarGlobal-18countries/scripts
try
    MainModel2_MakeFigures
    fprintf('FIGURES_DONE model 2\n');
catch ME
    fprintf(2, 'FIGURE SCRIPT ERROR (model 2): %s\n', getReport(ME));
end
exit
