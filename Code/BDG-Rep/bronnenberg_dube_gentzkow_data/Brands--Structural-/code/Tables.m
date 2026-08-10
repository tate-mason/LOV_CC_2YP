%=========================================================================
%=========================================================================
%
%                   TABLES.M
%
%=========================================================================
%=========================================================================
% FUNCTION:         Print estimation output to tables
% EXTERNAL CALLS:   fmincon (latest)
% NOTE:              
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%
%
%---------+---------+---------+---------+---------+---------+---------+---

clear all 
load ..\temp\results ;
load ..\temp\agedist 
load ..\temp\alphadelta  

%---------+---------+---------+---------+---------+---------+---------+---
%         table 6 
%---------+---------+---------+---------+---------+---------+---------+---

mpar6 = par_table6(1:2,1) ;
dpar6 = par_table6(1:2,2:end) ;
spar6 = std(dpar6,0,2) ;
tabletext = [mpar6(1) spar6(1) mpar6(2) spar6(2) keepavmu(1) ...
    std(keepavmu(2:end),0) log(.5)/log(mpar6(2,1)) keepfval(1)/1e05]' ; 

fid = fopen('..\output\tables.txt','a');
fprintf(fid, '<Tab:Struct>\n');
fclose(fid) ;
dlmwrite('..\output\tables.txt',tabletext,'-append','delimiter','\t')

%---------+---------+---------+---------+---------+---------+---------+---
%         table 8
%---------+---------+---------+---------+---------+---------+---------+---

%first add difference acros low and high ; then construct table
par_table8(5:6,:,:) = par_table8(1:2,:,:)-par_table8(3:4,:,:) ; 
mpar8 = squeeze(par_table8(:,1,:)); 
dpar8 = par_table8(:,2:end,:) ;
spar8 = squeeze(std(dpar8,0,2)) ;

tabletext = [mpar8([1 3 5],1)' mpar8([1 3 5],2)'; ...
             spar8([1 3 5],1)' spar8([1 3 5],2)'; ...
             mpar8([2 4 6],1)' mpar8([2 4 6],2)'; ...
             spar8([2 4 6],1)' spar8([2 4 6],2)'] ;
             
fid = fopen('..\output\tables.txt','a');
fprintf(fid, '<Tab:Struct_Split>\n');
fclose(fid) ;
dlmwrite('..\output\tables.txt',tabletext,'-append','delimiter','\t')

%---------+---------+---------+---------+---------+---------+---------+---
%         table appendix B
%---------+---------+---------+---------+---------+---------+---------+---
% This table lists all robustness checks. Each row reports estimates of  
% [alpha delta halflife years_to_convergence]. Standard errors are not
% reported but are stored in the matrix std_err.

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

%add "half life" and "time to convergence"

Lead = 10  ;             % lead in years
inv = [1 .35] ;           % supply side effort by 2nd entrant

for il = 1:size(estimates,1) ,
    alpha = estimates(il,1) ;
    delta = estimates(il,2) ;
    
    duration = Lead ;
    [y] = Get_Response(inv,duration,agedist,bins,alpha,delta) ;
    
    % find the #years that equate share
    keep = find(y(1:end-1)>.5 & y(2:end)<=.5) ;
    if 1-isempty(keep),
        ttc(il,1) = keep ;
    end
    
    hl(il,1) = log(.5)/log(delta) ;
    if hl(il,1) > 1000 , 
        hl(il,1) = NaN ;
    end
   
    il
end
% Legend to rows in estimates, ttc, and hl:
% 1 dependent variable: purch; no weights (base case)
% 2 dependent variable: equiv; no weights
% 3 dependent variable: expen; no weights
% 4 dependent variable: units; no weights
% 5 dependent variable: purch; weights
% 6 module state split: both brands available - no
% 7 module state split: both brands available - yes
% 8 module split: top two joint share - low
% 9 module split: top two joint share - high
%10 module split: purchase frequency - low
%11 module split: purchase frequency - high
%12 module split: geographic variation *) - low
%13 module split: geographic variation *) - high
%14 module split: geographic variation **) - low
%15 module split: geographic variation **) - high
%16 hhld split: gap = 0
%17 hhld split: gap > 0
%18 hhld split: gap <= 5 (should be the same as base case)
%19 hhld split: gap > 5
%notes: *)  state level shares computed from the sum across hh of 
%           purchases (this is the correct estimate of state level 
%           shares) 
%       **) state level shares computed from the average across hh 
%           of purchase shares (this is what we report in 
%           Appendix 2 as aggregate purchase share)

tabletext = [estimates ttc hl] ;
select = [1 2 3 4 5 7 16 ] ; 
fid = fopen('..\output\tables.txt','a');
fprintf(fid, '<Tab:Robustness>\n');
fclose(fid) ;
dlmwrite('..\output\tables.txt',tabletext(select,:), ...
         '-append','delimiter','\t')

select = [8 9 10 11 14 15] ; 
fid = fopen('..\output\tables.txt','a');
fprintf(fid, '<Tab:ExtraSplits>\n');
fclose(fid) ;
dlmwrite('..\output\tables.txt',tabletext(select,:), ...
         '-append','delimiter','\t')

%Tables for reviewer replies: 
%Rev 2 Point 2:
ingap = [16:19] ;
tt = [estimates(ingap,1) std_err(ingap,1) ...
      estimates(ingap,2) std_err(ingap,2)]' ; 
%---------+---------+---------+---------+---------+---------+---------+---
%         table appendix C
%---------+---------+---------+---------+---------+---------+---------+---
% This table lists alternative estimates of overlapping generations.
% Each row list 
% [alpha delta halflife years_to_convergence]. 
% Standard errors are not reported but are stored in the matrix std_err.

estimates = [] ;
std_err = [] ;

%add results from alternative degrees of generation-overlap 
mpar3 = squeeze(par_generations3(:,1,:)); 
dpar3 = par_generations3(:,2:end,:) ;
spar3 = squeeze(std(dpar3,0,2)) ;
          
for il = 1:size(mpar3,2) ,
    clear y k ;
    estimates = [estimates ; ...
                 [mpar3(1,il) mpar3(2,il) mpar3(5,il) ]] ;
    
    std_err = [std_err ; ...
                 [spar3(1,il) spar3(2,il) spar3(5,il) ]] ;
end


%add "half life" and "time to convergence"

Lead = 10  ;             % lead in years
inv = [1 .35] ;           % supply side effort by 2nd entrant
ttc = [] ;
hl = [] ;
for il = 1:size(estimates,1) ,
    alpha = estimates(il,1) ;
    delta = estimates(il,2) ;
    
    duration = Lead ;
    [y] = Get_Response(inv,duration,agedist,bins,alpha,delta) ;
    
    % find the #years that equate share
    keep = find(y(1:end-1)>.5 & y(2:end)<=.5) ;
    if 1-isempty(keep),
        ttc(il,1) = keep ;
    end
    
    hl(il,1) = log(.5)/log(delta) ;
    if hl(il,1) > 1000 , 
        hl(il,1) = NaN ;
    end
   
    il
end

%Legend to rows in estimates, ttc, and hl:
%1 initial endowment of 0 years of parents capital (base case)
%2 initial endowment of 10 years of parents capital 
%3 initial endowment of 20 years of parents capital
%4 initial endowment of 30 years of parents capital
%5 initial endowment of 40 years of parents capital
%6 initial endowment of 50 years of parents capital
%6 initial endowment of 60 years of parents capital

tabletext = [estimates(:,1:2) ttc hl] ;
select = [3 5 7] ; 
tabletext = tabletext(select,:) ;
fid = fopen('..\output\tables.txt','a');
fprintf(fid, '<Tab:Generations>\n');
fclose(fid) ;
dlmwrite('..\output\tables.txt',tabletext,'-append','delimiter','\t')

%table for reviewer 
w = [1 3 5 7] ; 
tt = [estimates(w,1) std_err(w,1) ...
      estimates(w,2) std_err(w,2) ...
      estimates(w,3) std_err(w,3) ...
      ]' ;




          