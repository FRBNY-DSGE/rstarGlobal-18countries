cd /data/dsge_data_dir/mdn/rstarGlobal-18countries/scripts
try
    MainModel3_B_MakeFigures
    fprintf('FIGURES_DONE model 3 B\n');
catch ME
    fprintf(2, 'FIGURE SCRIPT ERROR (model 3 B): %s\n', getReport(ME));
end
exit
