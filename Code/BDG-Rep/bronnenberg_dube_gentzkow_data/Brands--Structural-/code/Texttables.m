%=========================================================================
%=========================================================================
%
%                   TEXTTABLES.M
%
%=========================================================================
%=========================================================================
% FUNCTION:         File Undocumented Claims
% EXTERNAL CALLS:   
% NOTE:              
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------+---------+---------+---------+---------+---------+---------+---

clear all 

%---------+---------+---------+---------+---------+---------+---------+---
%         read results from estimation
%---------+---------+---------+---------+---------+---------+---------+---

load ..\temp\results ;

%---------+---------+---------+---------+---------+---------+---------+---
%         compute statistics to report
%---------+---------+---------+---------+---------+---------+---------+---
mpar = squeeze(par_table6(:,1)) ; %Npar by S


%---------+---------+---------+---------+---------+---------+---------+---
%         table <Tab:Demand_Dynamics>
%---------+---------+---------+---------+---------+---------+---------+---
tabletext = [] ;
tabletext = ...
    [mpar(1,1);
    mpar(1,1)*0.6+(1-mpar(1,1))*0.5];

fid = fopen('..\output\texttables.txt','a');
fprintf(fid, '<Tab:Struct>\n');
fclose(fid);
dlmwrite('..\output\texttables.txt',tabletext,'-append','delimiter','\t')

%---------+---------+---------+---------+---------+---------+---------+---
%         table <Tab:early_move_advantage>
%---------+---------+---------+---------+---------+---------+---------+---
clear all
load ..\temp\persistence.mat

tabletext=[];
tabletext=...
    [pr(1,1) pr(1,2)
    pr(2,1) pr(2,2)
    pr(3,1) pr(3,2)
    pr(4,1) pr(4,2)]
fid = fopen('..\output\texttables.txt','a');
fprintf(fid,'<Tab:early_move_advantage>\n');
fclose(fid);
dlmwrite('..\output\texttables.txt',tabletext,'-append','delimiter','\t')

%---------+---------+---------+---------+---------+---------+---------+---
%         standard errors for counterfactuals
%---------+---------+---------+---------+---------+---------+---------+---
load ..\temp\results ;
load ..\temp\agedist 
load ..\temp\alphadelta  

mpar6 = par_table6(1:2,1) ;
dpar6 = par_table6(1:2,2:end) ;
spar6 = std(dpar6,0,2) ;

% results with alternative dependent variables and weights 
mpar1 = squeeze(par_appendix1(1:2,1,1:4)); 
dpar1 = par_appendix1(1:2,2:end,1:4) ;
spar1 = squeeze(std(dpar1,0,2)) ;

estimates = [ mpar6(1,1) mpar6(2,1) ; ....
              mpar1(1,2) mpar1(2,2) ; ....
              mpar1(1,3) mpar1(2,3) ; ....
              mpar1(1,4) mpar1(2,4) ; ....
              mpar1(1,1) mpar1(2,1) ] ;

std_err = [   spar6(1,1) spar6(2,1) ; ....
              spar1(1,2) spar1(2,2) ; ....
              spar1(1,3) spar1(2,3) ; ....
              spar1(1,4) spar1(2,4) ; ....
              spar1(1,1) spar1(2,1) ] ;

          
%add results from alternative splits of the data 
mpar2 = squeeze(par_robust2(:,1,:)); 
dpar2 = par_robust2(:,2:end,:) ;
spar2 = squeeze(std(dpar2,0,2)) ;

for il = 1:size(mpar2,2) ,            
    for ie = 1:2 ,     
        estimates = [estimates ; ...
                     [mpar2((ie-1)*2+1,il) mpar2((ie-1)*2+2,il)] ] ;
        std_err    = [std_err ; ... 
                     [spar2((ie-1)*2+1,il) spar2((ie-1)*2+2,il)] ] ;
    end
end          

tabletext = [std_err] ;
select = [1 2 3 4 5 7 16 ] ; 
fid = fopen('..\output\texttables.txt','a');
fprintf(fid, '<Tab:Robustness_stderr>\n');
fclose(fid) ;
dlmwrite('..\output\texttables.txt',tabletext(select,:), ...
         '-append','delimiter','\t')
		 
		 