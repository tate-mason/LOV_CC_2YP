%=========================================================================
%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- First Stage Analysis
%
%=========================================================================
%=========================================================================
% FUNCTION:         Estimates first stage coefficients.
% DATA:           
%                   
% EXTERNAL CALLS:   none
% NOTE:             
%         
%---------+---------+---------+---------+---------+---------+---------+---
 
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------1---------2---------3---------4---------5---------6---------7---

clear all 

%---------+---------+---------+---------+---------+---------+---------+---
%         1.1         Read Data
%---------+---------+---------+---------+---------+---------+---------+---

lhs_raw = importdata('..\output\lhs_nonmig_bp.csv',',',1);
X_demo_raw = importdata('..\output\X_demo_nonmig.csv',',',1);
X_st_raw = importdata('..\output\X_st_nonmig.csv',',',1);

%---------+---------+---------+---------+---------+---------+---------+---
%         1.2         Preliminary data steps
%---------+---------+---------+---------+---------+---------+---------+---

X_hh = X_st_raw.data(:,1);
X_st = X_st_raw.data(:,2:end) ;
X_demo = X_demo_raw.data(:,2:end) ;

yhh = lhs_raw.data(:,1) ;
ymod = lhs_raw.data(:,2) ;
ypair = lhs_raw.data(:,3) ; 
q1 = lhs_raw.data(:,4) ;
q2 = lhs_raw.data(:,5) ;
y = q1./(q1+q2) ;

N = length(y) ; 
umod = unique(ymod) ; 
uhh = unique(yhh) ;
nm = length(umod) ;
nhh = length(uhh) ;
ns = size(X_st,2);
nd = size(X_demo,2);
upair = unique(ypair) ; 
np = length(upair) ;

%---------+---------+---------+---------+---------+---------+---------+---
%         1.4         Trap errors
%---------+---------+---------+---------+---------+---------+---------+---

if sum(ismember(X_hh,yhh))~=nhh 
    error('Number of HHs in lhs is not a subset of HHs in X matrices')
end

if X_demo_raw.data(:,1)~=X_st_raw.data(:,1)
    error('HH indices in X_demo and X_st do not match')
end

%---------+---------+---------+---------+---------+---------+---------+---
%         2         Estimation
%---------+---------+---------+---------+---------+---------+---------+---

b = NaN(np,ns+nd) ;
bnd = NaN(np,ns) ;

for i = 1:np 
    
    disp(['Current pair: ' num2str(upair(i))]);

    % select lhs data for pair i
    which = find(ypair==upair(i)) ; 
    y_i = y(which,:) ;
    yhh_i = yhh(which,:) ;
    
    % confirm that there are no duplicate values in yhh_i or X_hh
    if length(unique(yhh_i))<length(yhh_i) | ...
            length(unique(X_hh))<length(X_hh)
        error(['Duplicate hhs for module ' i])
    end
    
    % select rhs data for module i
    which = ismember(X_hh,yhh_i) ;
    X_st_i = X_st(which,:);
    X_demo_i = X_demo(which,:);
    X_i = [X_st_i X_demo_i];
    
    %check conditioning of the X_i matrix -- brute-force to warrant against
    %  dependencies for low-count states
    incl = [1:ns+nd];
    inclst = [1:ns];
    if  condest(X_i'*X_i)>1E12 , %drop demographics
       incl = find(std(X_i)>0) ;
       inclst = incl(1,1:(end-nd));
       X_i = X_i(:,incl);
       X_st_i = X_st_i(:,inclst) ;
    end
    
    bndtemp = (X_st_i'*X_st_i)\(X_st_i'*y_i) ;
    bnd(i,inclst) = bndtemp';
    btemp = (X_i'*X_i)\(X_i'*y_i) ;
    b(i,incl) = btemp' ;

    
end

%---------+---------+---------+---------+---------+---------+---------+---
%         3         Write output
%---------+---------+---------+---------+---------+---------+---------+---

b_st_header = ['brandpair' X_st_raw.colheaders(1,2:end)] ;
format = [repmat(['%s,'],1,size(b_st_header,2)-1) '%s\n'] ;
bnd_st = [upair bnd(:,1:ns)] ; 
fid = fopen('..\output\bnd_st_bp.csv','w') ;
fprintf(fid,format,b_st_header{1,:}) ;
fclose(fid) ;
dlmwrite('..\output\bnd_st_bp.csv',bnd_st,'precision',8,'-append') ;

b_st = [upair b(:,1:ns)] ;
fid = fopen('..\output\b_st_bp.csv','w') ;
fprintf(fid,format,b_st_header{1,:}) ;
fclose(fid) ;
dlmwrite('..\output\b_st_bp.csv',b_st,'precision',8,'-append') ;

b_demo_header = ['brandpair' X_demo_raw.colheaders(1,2:end)] ;
format = [repmat(['%s,'],1,size(b_demo_header,2)-1) '%s\n'] ;
b_demo = [upair b(:,(ns+1):(ns+nd))] ;
fid = fopen('..\output\b_demo_bp.csv','w') ;
fprintf(fid,format,b_demo_header{1,:}) ;
fclose(fid) ;
dlmwrite('..\output\b_demo_bp.csv',b_demo,'precision',8,'-append');





