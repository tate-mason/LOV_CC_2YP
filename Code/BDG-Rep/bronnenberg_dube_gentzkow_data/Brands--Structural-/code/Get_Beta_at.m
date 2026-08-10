function beta = Get_Beta_at(a,t, alpha, dd)  
%=========================================================================
%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- STRUCTURAL MODEL
%                   COMPUTATION OF BETA(A,T)
%
%=========================================================================
%=========================================================================
% FUNCTION:         This code computes the function beta(a,t,alpha,delta)
% EXTERNAL CALLS:   none
% NOTE:             a = current age, 
%                   t = number of years in current area of residence  
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------+---------+---------+---------+---------+---------+---------+---
%---------+---------+---------+---------+---------+---------+---------+---

beta = 1-(1-alpha)*prod( 1-alpha./dd(a+1:a+t-1) ) ;

%note: this equals alpha if t=1, as desired.  
%note: the elements into dd run from a+1 to a+t-1. 
%      this contains sum_{k=0}^{x} delta^k with x running from a to a+t-2  