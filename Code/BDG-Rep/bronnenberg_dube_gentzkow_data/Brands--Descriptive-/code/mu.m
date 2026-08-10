%=========================================================================
%
%    mu.m -- Compute beta_hat from input matrices
%
%=========================================================================

%=========================================================================
%
%    I. Analysis of modules
%
%=========================================================================

clear all 
addpath('../external/');

%---------+---------+---------+---------+---------+---------+---------+---
%         I.1         Read Data
%---------+---------+---------+---------+---------+---------+---------+---
b_demo = dlmread('..\external\b_demo_purch.csv',',',1,0);
b_st = dlmread('..\external\b_st_purch.csv',',',1,0);
bnd_st = dlmread('..\external\bnd_st_purch.csv',',',1,0);
lhs = dlmread('..\external\lhs_mig.csv',',',1,0);
X_demo = dlmread('..\external\X_demo_mig.csv',',',1,0);
X_stb = dlmread('..\external\X_stb_mig.csv',',',1,0);
X_stc = dlmread('..\external\X_stc_mig.csv',',',1,0);

index = lhs(:,1:2);

%---------+---------+---------+---------+---------+---------+---------+---
%         I.2         Compute Mu for model w/ no demos
%---------+---------+---------+---------+---------+---------+---------+---
bnd_demo = [b_demo(:,1) zeros(size(b_demo(:,2:end)))];
[mundb mundc] = computemu(index,bnd_demo,bnd_st,X_demo,X_stb,X_stc);

%---------+---------+---------+---------+---------+---------+---------+---
%         I.3         Compute Mu for main model
%---------+---------+---------+---------+---------+---------+---------+---
[mub muc] = computemu(index,b_demo,b_st,X_demo,X_stb,X_stc);

%---------+---------+---------+---------+---------+---------+---------+---
%         I.4         Write to /temp/
%---------+---------+---------+---------+---------+---------+---------+---
fid = fopen('..\temp\mu.csv','w') ;
fprintf(fid,'hhld_id,module,mundb,mundc,mub,muc\n') ;
fclose(fid) ;
dlmwrite('..\temp\mu.csv',[index mundb mundc mub muc],'-append','precision',10);

%=========================================================================
%
%    II. Analysis of brand pairs
%
%=========================================================================


clear all 
addpath('../external/');

%---------+---------+---------+---------+---------+---------+---------+---
%         II.1         Read Data
%---------+---------+---------+---------+---------+---------+---------+---
b_demo = dlmread('..\external\b_demo_bp.csv',',',1,0);
b_st = dlmread('..\external\b_st_bp.csv',',',1,0);
bnd_st = dlmread('..\external\bnd_st_bp.csv',',',1,0);
lhs = dlmread('..\external\lhs_mig_bp.csv',',',1,0);
X_demo = dlmread('..\external\X_demo_mig.csv',',',1,0);
X_stb = dlmread('..\external\X_stb_mig.csv',',',1,0);
X_stc = dlmread('..\external\X_stc_mig.csv',',',1,0);

index = lhs(:,[1 3]);

%---------+---------+---------+---------+---------+---------+---------+---
%         II.2         Compute Mu for model w/ no demos
%---------+---------+---------+---------+---------+---------+---------+---
bnd_demo = [b_demo(:,1) zeros(size(b_demo(:,2:end)))];
[mundb mundc] = computemu(index,bnd_demo,bnd_st,X_demo,X_stb,X_stc);

%---------+---------+---------+---------+---------+---------+---------+---
%         II.3         Compute Mu for main model
%---------+---------+---------+---------+---------+---------+---------+---
[mub muc] = computemu(index,b_demo,b_st,X_demo,X_stb,X_stc);

%---------+---------+---------+---------+---------+---------+---------+---
%         II.4         Write to /temp/
%---------+---------+---------+---------+---------+---------+---------+---
fid = fopen('..\temp\mu_bp.csv','w') ;
fprintf(fid,'hhld_id,pairid,mundb,mundc,mub,muc\n') ;
fclose(fid) ;
dlmwrite('..\temp\mu_bp.csv',[index mundb mundc mub muc],'-append','precision',10);

