
function [share, capital] = ...
    Get_Response(inv, duration, agedist, bins, alpha, delta, e)  
%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- COUNTERFACTUALS RESPONSE
%                                             PREDICTION
%
%=========================================================================
% FUNCTION:         Computes the market share and brand capital in a market
%                   consisting of a mix of householdheads of different age
%                   cohorts. 
% DATA:             invest_shares = fraction of marketing budget invested 
%                   duration      = [#lead periods, #catch up periods] 
%                                   (# years of years investing at inv1
%                                   followed by # years investing at inv2 
%                   agedist       = shares of age distribution
%                   alpha, delta  = parameters                  
% EXTERNAL CALLS:   none
% NOTE:            
%  (1) note caps on effective error draws to not let m and y stray
%      out of [0,1]
%
%  (2) the marketing effort is equal to market share: 
%         yt = alpha yt + (1-alpha) integral_i kit + et  (or equivalently)
%         yt = integral_i kit + et/(1-alpha) 
%  
%  (3) share y and capital k are objects organized by calendar year               
%                   and age in a given calendar year. 
% 
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------+---------+---------+---------+---------+---------+---------+---

maxT = 100 ; 
maxAge = 100 ; %max age

y = inv(1)*ones(maxAge,duration(1)+maxT) ; 
k = inv(1)*ones(maxAge+1,duration(1)+maxT) ; 
 

for t = 1:maxT ,
    
    T = duration(1)+t ;   %calendar years
    
    if nargin==7 ,
        mutemp = agedist'*k(bins,T) + e(t)/(1-alpha) ;%see note (2)
        mu = min(max(mutemp,0),1) ;                   %see note (1)
    else 
        mu = inv(2) ; 
    end
    
    k(1,T) = mu ;         %brand stock of a one-year old

    for a = 1:maxAge      %age in T
        treatment_age = min(a,T) ;
        series = 1:treatment_age ;
        
        if nargin==7 ,
            ytemp = alpha*mu+(1-alpha)*k(a,T) + e(t) ;
            y(a,T) = min(max(ytemp,0),1) ;  %see note (1)
        else
            y(a,T) = alpha*mu+(1-alpha)*k(a,T) ;
        end
        
        history = sub2ind(size(y),a+1-series,T+1-series) ;
        k(a+1,T+1) = sum(y(history).*(delta.^(series)))/ ...
            sum(delta.^(series)) ;
    end
    
    %the capital in the last age group (maxAge+1) expires
    
end

share = sum(y(bins,end-maxT+1:end).*repmat(agedist,1,maxT)) ;
capital = sum(k(bins,end-maxT+1:end).*repmat(agedist,1,maxT)) ;
