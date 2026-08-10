function [mub,muc] = computemu(index,b_demo,b_st,X_demo,X_stb,X_stc)

%=========================================================================
%
%    computemu.m -- Compute mu from input matrices
%
%=========================================================================
%=========================================================================
% FUNCTION:         This function computes mu for each hh-module pair
%                   using vectors of parameter estimates and data matrices
%                   as inputs
%
% nm = number of modules
% nh = number of households
% ns = number of states
% nd = number of demographic variables
%
% INPUTS:
%   index       nm*nh x 2     Household-module indices for ouput. First
%                             col gives hh indices. Second gives module
%                             indices.
%   b_demo      nm x nd       Coeffs on demographics (one row per mod)
%   b_st        nm x ns       Coeffs on state dummies (one row per mod)
%   X_demo      nh x nd       Demographic data (one row per hh)
%   X_stb       nh x ns       State of birth dummies (one row per hh)
%   X_stc       nh x ns       Current state dummies (one row per hh)
%
%---------+---------+---------+---------+---------+---------+---------+---

%---------+---------+---------+---------+---------+---------+---------+---
%         1         Format Data
%---------+---------+---------+---------+---------+---------+---------+---
hh = index(:,1);
mod = index(:,2);
Xhh = X_demo(:,1);
bmod = b_demo(:,1);

Xb = [X_demo(:,2:end) X_stb(:,2:end)];
Xc = [X_demo(:,2:end) X_stc(:,2:end)];
b = [b_demo(:,2:end) b_st(:,2:end)];

%---------+---------+---------+---------+---------+---------+---------+---
%         2         Expand X and b matrices
%---------+---------+---------+---------+---------+---------+---------+---
[tf,xindex] = ismember(hh,Xhh);
[tf,bindex] = ismember(mod,bmod);
Xb = Xb(xindex,:);
Xc = Xc(xindex,:);
b = b(bindex,:);

%---------+---------+---------+---------+---------+---------+---------+---
%         3         Compute mub and muc
%---------+---------+---------+---------+---------+---------+---------+---

% flag hh-module pairs where b=NaN and corresponding X is non-zero
% (these are cases where there were no non-migrants in the
% module-state cell in question but there are migrants in that
% cell for either birth or current state)
flagb = max(isnan(b) & (Xb~=0),[],2);
flagc = max(isnan(b) & (Xc~=0),[],2);

% change NaN elements of b to zero
b(isnan(b)) = 0;

% compute mu
mub = sum(Xb.*b,2);
muc = sum(Xc.*b,2);

% replace flagged cases with NaN
mub(flagb) = NaN;
muc(flagc) = NaN;



