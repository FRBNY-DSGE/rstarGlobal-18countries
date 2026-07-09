cd /data/dsge_data_dir/mdn/rstarGlobal-18countries/scripts
try
    MainModel1_MakeFigures
    fprintf('FIGURES_DONE model 1\n');
catch ME
    fprintf(2, 'FIGURE SCRIPT ERROR (model 1): %s\n', getReport(ME));
end
exit
