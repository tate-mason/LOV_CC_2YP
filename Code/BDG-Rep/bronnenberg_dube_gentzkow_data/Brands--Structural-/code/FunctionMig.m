function [func, bat, avmu, p, funcval] = FunctionMig(par)
%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- STRUCTURAL MODEL
%                   COMPUTATION OF OBJECTIVE FUNCTION
%
%=========================================================================
% FUNCTION:         This code computes the objective function for the
%                   structural model in the paper according to several
%                   specifications indexed by the variable spec
%
% DATA:             globals 
%
% EXTERNAL CALLS:   Get_Beta_at
%
% NOTE:              
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------+---------+---------+---------+---------+---------+---------+---


%---------+---------+---------+---------+---------+---------+---------+---
%         constants and data
%---------+---------+---------+---------+---------+---------+---------+---
global AGE YEARS XBMU XDMU Q MODIFIER WEIGHT ;

N = length(YEARS) ; % number of observations in the data

partrans = exp(par)./(1+exp(par)) ; % structural parameters between 0-1
alpha(1,1) = partrans(1) ;        
delta(1,1) = partrans(2) ; 
alpha(2,1) = partrans(3) ; 
delta(2,1) = partrans(4) ; 


%---------+---------+---------+---------+---------+---------+---------+---
%         Compute brand stock for each age a and number of years
%         t you live in current territory
%---------+---------+---------+---------+---------+---------+---------+---

LevMod = max(MODIFIER)+1 ;              % # levels in MODIFIER     
m = min(AGE) ;                          %  
M = max(AGE) ;                          %
bat = [] ;                              % 3-d array of predicted beta
dd = [cumsum(delta(1).^([0:1:M])') ...
      cumsum(delta(2).^([0:1:M])')] ; 

% compute predicted beta across modifier, agemove, yearssincemove

for a = 1:M-1 , %a = years in state of birth
    for t = 1:M-a , %t = years in state of residence
        for c = 1:LevMod ,
            bat(a,t,c) = Get_Beta_at(a,t, alpha(c), dd(:,c)) ;
        end
    end
end

%---------+---------+---------+---------+---------+---------+---------+---
%         Select from bat the observation located in the column t=years, 
%         the row a = age at move, and the level of the modifier
%---------+---------+---------+---------+---------+---------+---------+---
YEARS = max(1,YEARS) ; % minimum number of years in current state is one
agemove = max(AGE-YEARS,1) ; %minimum number of years in birth state is one
index = sub2ind(size(bat),agemove,YEARS,MODIFIER+1);  

%extract predicted beta from bat
beta_hat = bat(index) ;

%---------+---------+---------+---------+---------+---------+---------+---
%         Compute goal function over top 2 brands
%---------+---------+---------+---------+---------+---------+---------+---

%predicted shares for migrants 
p = (XBMU + beta_hat.*XDMU) ;
select = isnan(Q.*XBMU.*XDMU)==0 ; %see flagged cases in computemu.m
part = (WEIGHT.*((p-Q).^2)) ;
func = sum(part(select))/sum(select) ;
avmu = mean(XBMU(select)+XDMU(select)) ;
funcval = full(func*sum(select)) ;
func = full(func) ;