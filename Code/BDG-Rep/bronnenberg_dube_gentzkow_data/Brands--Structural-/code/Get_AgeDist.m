
function [agedist] = Get_AgeDist(bins)  
%=========================================================================
%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- COMPUTE PANEL AGE DIST
%
%=========================================================================
%=========================================================================
% FUNCTION:         Compute the age dist of panelists
% DATA:             homescan records
% EXTERNAL CALLS:   none
% NOTE:              
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------+---------+---------+---------+---------+---------+---------+---

%---------1---------2---------3---------4---------5---------6---------7---
% CONTENTS
%---------+---------+---------+---------+---------+---------+---------+---


hh_char = importdata('..\temp\hh_char.csv',',',1);
N = hist(hh_char.data(:,2),bins) ;
agedist = N/sum(N) ;  
