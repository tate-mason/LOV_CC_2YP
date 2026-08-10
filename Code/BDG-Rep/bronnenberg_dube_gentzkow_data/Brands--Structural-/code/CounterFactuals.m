%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- COUNTERFACTUALS
%
%=========================================================================
% FUNCTION:         This code creates counter factual computations.
% DATA:             Estimated values for selected parameters. 
% EXTERNAL CALLS:   Get_AgeDist 
%                   Get_Response
% NOTE:             block 1: persistence
%                   block 2: time until convergence
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------+---------+---------+---------+---------+---------+---------+---


%%
%---------+---------+---------+---------+---------+---------+---------+---
%         1         persistence graph
%---------+---------+---------+---------+---------+---------+---------+---

addpath('..\external\')
clear all
rand('seed',25041963) ;
load ..\temp\alphadelta  ;

bins = [18:1:95]' ; 
agedist = Get_AgeDist(bins)';

%---------+---------+---------+---------+---------+---------+---------+---
%  Sample paths
%  Notes:
%      share contains the market shares at year t 
%---------+---------+---------+---------+---------+---------+---------+---
rounddelta = .001*round(1000*delta) ; 
duration = 100 ; %initialization years without shocks
inv = [.75 ] ; %prior investment (and market shares). 
shock = .05 ; 
R = 1000 ;% test at lower values, e.g., 50. Set to 1000 for final draft... 
keep = zeros(4,R); 
T = 100 ;
for d = 1:4
    delta = rounddelta -.25*(d-1) ; 
    for r = 1:R ;                          %realizations
        e = 2*shock*(rand(T,1)-.5) ;       %market level shocks 
        share = Get_Response(inv, duration, agedist, bins, alpha, delta, e) ;
        keep(d,r) = share(100) ;
    end
end

%---------+---------+---------+---------+---------+---------+---------+---
%         make graph 
%---------+---------+---------+---------+---------+---------+---------+---

set(0, 'defaultaxesfontsize',9) ; %default font size
w= .7 ;  %spacing constant

for j = 1:size(keep,1) ,
    d = rounddelta-.25*(j-1) ;
    subplot(2,2,j) ;
    nbins = [0:.1:1.01] ;
    N = hist(keep(j,:),nbins) ;
    prob = N/sum(N) ;
    bar(nbins,prob, w); colormap(repmat([0.7 0.7 0.7],64,1));
    axis([-.05 1.05 0 .53]) ;
    ti = strcat('delta= ',' ',num2str(d)) ;
    title(char(ti))
    xlabel('Share')
    
    halflife(j,1) = log(.5)/log(d) ;

    for q = 1:2 , %number of cases within +/- 10 or 20% of initial
       retain(q) =length(find(keep(j,:)<inv(1)+q*.1 & keep(j,:)>inv(1)-q*.1));
    end ;
    pr(j,:) = round(retain/(R/100))/100 ;  
   
end
%---------+---------+---------+---------+---------+---------+---------+---
%         1.4                 write 
%---------+---------+---------+---------+---------+---------+---------+---

print -deps -r600 -painters '..\output\figures\persistence.eps'
save '..\temp\agedist.mat' agedist bins; 
save '..\temp\persistence.mat' pr halflife;


%%
%---------+---------+---------+---------+---------+---------+---------+---
%         2          early mover advantage
%
% The code below computes the number of years a 2nd entrant needs to 
% invest to achieve equal shares given a level of investment.
%
% The age distribution of primary shoppers enters the counterfactual
% computations because consumers can not be treated more years than age.
% Primary shoppers are all 18+ years of age. 
%
% Computation of age distribution is currently done using empirical
% distribution of age of primary shoppers in Nielsen panel. 
% This age distribution of primary shoppers may not be representative. 
% This seems a minor point, but if so desired we could use the census.
%---------+---------+---------+---------+---------+---------+---------+---

clear all 
addpath('..\external\')
load ..\temp\agedist 
load ..\temp\alphadelta  

% elast. of substitution parameter from /analysis/Brands (IRI Elasticities)
%b = loadparam('coeff','..\external\elast_of_subs.txt');
%Lead = [1 5 10 15 25 ] ;             % lead in years
%mu_2nd = [.50 .55 .6 .65 .7 .75 ] ;       % supply side effort by 2nd entrant
%discount = 1 - (...
%                (alpha*mu_2nd    +0.5*(1-alpha))./ ...
%                (alpha*(1-mu_2nd)+0.5*(1-alpha))  ...
%                ).^(1/b) ;           % equivalent discount


% Price Effect from/analysis/Brands (IRI Elasticities)
b1 = loadparam('coeff','..\external\linprob_price_effect.txt');
mu_price = b1/alpha ;
Lead = [1 5 10 15 25 ] ;			% lead in years
mu_2nd = [.50 .55 .6 .65 .7 .75 ] ;		% supply side effort by 2nd entrant
discount = 1 - exp( (mu_2nd-0.5)/mu_price )	% equivalent discount


InvestY = NaN(length(Lead),length(mu_2nd)-1) ;
for il = 1:length(Lead) ,            %lead length
    for ie = 1:length(mu_2nd) ,      %effort
        clear y k ;
        inv = [1 1-mu_2nd(ie)] ;
        duration = Lead(il) ;
        [y, k] = Get_Response(inv,duration,agedist,bins,alpha,delta) ;
               
        % find the #years that equate share
        indexY = find(y(1:end-1)>.5 & y(2:end)<=.5) ;
        if ie>1 ,
            InvestY(il,ie-1) = indexY ;
        end
        keepY(:,il,ie) = 1-y' ; 
    end
end

save '..\temp\cfac.mat' InvestY discount ;
fid = fopen('..\output\tables.txt','a');
fprintf(fid, '<Tab:Struct_Catchup>\n');
fclose(fid) ;
dlmwrite('..\output\tables.txt',round(100*discount(2:end)),'-append','delimiter','\t')
dlmwrite('..\output\tables.txt',InvestY,'-append','delimiter','\t')

%---------+---------+---------+---------+---------+---------+---------+---
%figure of keepY
%---------+---------+---------+---------+---------+---------+---------+---

figure
T = size(keepY,1) ;

for i = 1:size(keepY,2) ;
    hold on;
    
    plot([1:T]',squeeze(keepY(1:T,i,1)),'k-')
end
axis([1 50 .3 .5])
ti = strcat('\mu 2nd entrant= ',' ',num2str(mu_2nd(1))) ;
text(17.5,0.68,ti,'HorizontalAlignment','Center') ;
text(15, .37, '25 year lag') ;
text(13.75, .38, '15 year lag') ;
text(12.5, .39, '10 year lag') ;
text(10, .41, '5 year lag') ;
text(7.5, .43, '1 year lag') ;
xlabel('years in market')

print -deps -r600 -painters '..\output\figures\dynamics_catchup.eps'

hold off ;


%---------+---------+---------+---------+---------+---------+---------+---
% Reported dynamics in counterfactual section of the paper.
%
% Dyn_share and dyn_stock give the dynamics of share and stock from 
% a change in investment from 0.5 to 0.6 depending on the number of 
% years that you invest
%
% Dyn_response is the immediate response to an increase in investment to .6
% and the response to the retraction of that investment one year later 
%---------+---------+---------+---------+---------+---------+---------+---

invest_shares = [.5 .6] ;
initial = 100 ; %number of years on the market prior to mu changing 
duration = initial  ;
[dyn_share dyn_stock] = ...
        Get_Response(invest_shares, ...
        duration, agedist, bins,alpha,delta) ;
dyn_response(1,1) = dyn_share(1) ; 
dyn_response(2,1) = alpha*invest_shares(1) + (1-alpha)*dyn_stock(2) ;

save '..\temp\cfdyn.mat' dyn_share dyn_stock dyn_response;



