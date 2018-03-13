% check whther there are any NaNs in mean and std files
close all;
clear all;
nmovavg = 100; % moving average size
namedir = '../RESULTS/CBRAIN-nolat-nonorm/csv_files';
%namedir = '../CBRAIN-pg-dev-nolat-batch256-nonormout/csv_files';
max_time = 1;
 
cost_functions_plot = {'loss/RMSE_0';'loss/logRMSE_0';'loss/mse_0';'loss/logloss_0';'loss/absloss_0';'loss/R2_0';'loss/mape_0';'loss/loss_0';...
                       'lossAvgLev/R2AvgLev/0_0';'lossAvgLev/R2AvgLev/1_0';'lossAvgLev/R2AvgLev/2_0';'lossAvgLev/R2AvgLev/3_0';...
                       'lossAvgLev/R2AvgLev/4_0';'lossAvgLev/R2AvgLev/5_0';'lossAvgLev/R2AvgLev/6_0';'lossAvgLev/R2AvgLev/7_0';...
                       'lossAvgLev/R2AvgLev/8_0';'lossAvgLev/R2AvgLev/9_0';'lossAvgLev/R2AvgLev/10_0';'lossAvgLev/R2AvgLev/11_0';...
                       'lossAvgLev/R2AvgLev/12_0';'lossAvgLev/R2AvgLev/13_0';'lossAvgLev/R2AvgLev/14_0';'lossAvgLev/R2AvgLev/15_0';...
                       'lossAvgLev/R2AvgLev/16_0';'lossAvgLev/R2AvgLev/17_0';'lossAvgLev/R2AvgLev/18_0';'lossAvgLev/R2AvgLev/19_0';...
                       'lossAvgLev/R2AvgLev/20_0';...
                       'lossAvgVar/R2AvgVar/TPHYSTND_NORAD_0';...
                       'lossAvgVar/R2AvgVar/PHQ_0'};
for i=1:length(cost_functions_plot)
    name = cost_functions_plot{i};
    cost_functions_name{i} = name(1:end-2);
end
% saving table of results
counter = 0;
listing = dir(namedir);
Cost = [];
RunName = [];
for i=1:length(listing)
    if(length(listing(i).name)>4)
       if(strcmp(listing(i).name(end-2:end),'csv'))
           filename = [namedir  '/' listing(i).name]
           %M = tblread(filename,',');
           M = csvimport(filename);
           header = M(1,:);
           M = M(2:end,:);
           time = 1:length(M(:,1));
           if(length(time)>2*nmovavg)
               counter = counter + 1;
               layersbeg = strfind(listing(i).name,'layers_');
               layersend = strfind(listing(i).name,'_lr');
               layers{counter}    = listing(i).name(layersbeg+7:layersend-1);
               acbeg = strfind(listing(i).name,'_ac_');
               acend = strfind(listing(i).name,'_conv');
               ac{counter}    = listing(i).name(acbeg+4:acend-1);
               lrbeg = strfind(listing(i).name,'_lr_');
               lrend = strfind(listing(i).name,'_lrstep');
               learnrate(counter)    = str2num(listing(i).name(lrbeg+4:lrend-1));
               convbeg = strfind(listing(i).name,'_conv_');
               convend = strfind(listing(i).name,'_locconv');
               conv{counter}    = listing(i).name(convbeg+6:convend-1);
               lossbeg = strfind(listing(i).name,'_loss_');
               lossend = strfind(listing(i).name,'.csv');
               loss{counter}    = listing(i).name(lossbeg+6:lossend-1);
               locconvbeg = strfind(listing(i).name,'_locconv_');
               locconvend = strfind(listing(i).name,'_vars');
               locconv{counter}    = listing(i).name(locconvbeg+9:locconvend-1);


               % plot statistics
               vect_cost = [];
               figure(1)
               for ii=1:length(cost_functions_plot)
                    index = find(strcmp(cost_functions_plot{ii},header)>0);
                    if(ii<=8)
                        subplot(2,4,ii)
                        plot(time,cell2mat(M(:,index)))
                        hold all;
                        kk = strfind(cost_functions_plot{ii},'R2'); 
                        max_time = max(time(end), max_time);
                        set(gca, 'XLim', [0 max_time]);
                        if(~isempty(kk))
                            set(gca, 'YLim', [0 1]);
                        end
                        title(cost_functions_plot{ii}) 
                    end
                    mov = movmean(cell2mat(M(1:end-1,index)),nmovavg);
                    vect_cost(ii) = mov(end-nmovavg);
               end

               RunName{counter} = listing(i).name;
               Cost = [Cost;vect_cost];
           end
       end
    end 
end 
savefig(gcf,[namedir '/loss_compare.fig']);

varnames = {'layers','activation','convo','local_convo','learnrate','loss_fct'};
varnames(7:7+31-1) = cost_functions_name;
T = table(layers',ac',conv',locconv',learnrate',loss',...
    Cost(:,1),Cost(:,2),Cost(:,3),Cost(:,4),Cost(:,5),...
    Cost(:,6),Cost(:,7),Cost(:,8),Cost(:,9),Cost(:,10),...
    Cost(:,11),Cost(:,12),Cost(:,13),Cost(:,14),Cost(:,15),...
    Cost(:,16),Cost(:,17),Cost(:,18),Cost(:,19),Cost(:,20),...
    Cost(:,21),Cost(:,22),Cost(:,23),Cost(:,24),Cost(:,25),...
    Cost(:,26),Cost(:,27),Cost(:,28),Cost(:,29),Cost(:,30),Cost(:,31),...
    'RowNames',RunName);
varnames = strrep(varnames,'/','_');
T.Properties.VariableNames = varnames;
writetable(T,[namedir '/statistics.xls']);
% %);%)

'end' 