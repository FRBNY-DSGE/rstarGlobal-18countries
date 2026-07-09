cd /data/dsge_data_dir/mdn/rstarGlobal-18countries/scripts
try
    MainModel3_MakeFigures
    fprintf('FIGURES_DONE model 3\n');
catch ME
    fprintf(2, 'FIGURE SCRIPT ERROR (model 3): %s\n', getReport(ME));
end
exit
