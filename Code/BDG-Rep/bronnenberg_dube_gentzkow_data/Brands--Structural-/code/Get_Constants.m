%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- Get_Contstants
%
%=========================================================================
% FUNCTION:         Loads selected constants
%
% DATA:             
%
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------+---------+---------+---------+---------+---------+---------+---


s = RandStream.create('mt19937ar','Seed',25041963) ; 
% to replicate bootstrap samples

R = 26 ;  % # bootstrap samples + 1
          
keepscale = 10E05 ; % order of magnitude of the goal function
          
npar = 4 ; % number of parameters
          
nm = 238 ; % number of modules

options=optimset(...    
    'TolFun',1e-6,...
    'MaxFunEvals',20000,...
    'Diagnostics', 'on',...
    'LargeScale','off',...
    'Display','iter') ; %optimization tolerances and settings

for rep = 1:R %draw indeces for bootstrap samples
    if rep>1 , 
        index = randi(s,nm,nm,1) ; %generate bootstrap index
        keepmodind(:,rep) = index ; 
    else   %rep=1
        keepmodind(:,rep) = [1:nm]' ; % result is nm by R
    end ; 
end ; 
