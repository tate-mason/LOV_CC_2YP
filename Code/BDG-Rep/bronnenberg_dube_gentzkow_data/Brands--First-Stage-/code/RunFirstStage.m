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

lhs_raw = importdata('..\output\lhs_nonmig.csv',',',1);
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
N = size(lhs_raw.data,1) ;
umod = unique(ymod) ;
uhh = unique(yhh) ;
nm = length(umod) ;
nhh = length(uhh) ;
ns = size(X_st,2);
nd = size(X_demo,2);

for jc=1:4 ,
    switch jc ,
        case 1, lab = 'purch' ;
        case 2, lab = 'equiv' ;
        case 3, lab = 'exp' ;
        case 4, lab = 'unit' ;
    end
            
    q1 = lhs_raw.data(:,1+2*jc) ;
    q2 = lhs_raw.data(:,2+2*jc) ;
    y = q1./(q1+q2) ;
    
%---------+---------+---------+---------+---------+---------+---------+---
%         1.4         Trap errors
%---------+---------+---------+---------+---------+---------+---------+---
    
    if size(X_st,1)~=nhh | size(X_demo,1)~=nhh
        error('Number of HHs in lhs and X matrices do not match')
    end
    
    if X_demo_raw.data(:,1)~=X_st_raw.data(:,1)
        error('HH indices in X_demo and X_st do not match')
    end
    
%---------+---------+---------+---------+---------+---------+---------+---
%         2         Estimation
%---------+---------+---------+---------+---------+---------+---------+---
    
    b = NaN(nm,ns+nd) ;
    bnd = NaN(nm,ns) ;
    
    for i = 1:nm
        
        disp(['Current module: ' num2str(umod(i)) lab]);
        
        % select lhs data for module i
        which = find(ymod==umod(i) & q1+q2>0) ;
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
    
    b_st_header = ['module' X_st_raw.colheaders(1,2:end)] ;
    format = [repmat(['%s,'],1,size(b_st_header,2)-1) '%s\n'] ;
    
    bnd_st = [umod bnd(:,1:ns)] ;
    fn = strcat('..\output\bnd_st_',lab,'.csv') ;
    fid = fopen(fn,'w') ;
    fprintf(fid,format,b_st_header{1,:}) ;
    fclose(fid) ;
    dlmwrite(fn,bnd_st,'-append') ;
    
    b_st = [umod b(:,1:ns)] ;
    fn = strcat('..\output\b_st_',lab,'.csv') ;
    fid = fopen(fn,'w') ;
    fprintf(fid,format,b_st_header{1,:}) ;
    fclose(fid) ;
    dlmwrite(fn,b_st,'-append') ;
    
    b_demo_header = ['module' X_demo_raw.colheaders(1,2:end)] ;
    format = [repmat(['%s,'],1,size(b_demo_header,2)-1) '%s\n'] ;
    b_demo = [umod b(:,(ns+1):(ns+nd))] ;
    fn = strcat('..\output\b_demo_',lab,'.csv') ;
    fid = fopen(fn,'w') ;
    fprintf(fid,format,b_demo_header{1,:}) ;
    fclose(fid) ;
    dlmwrite(fn,b_demo,'-append');
    
end ;



