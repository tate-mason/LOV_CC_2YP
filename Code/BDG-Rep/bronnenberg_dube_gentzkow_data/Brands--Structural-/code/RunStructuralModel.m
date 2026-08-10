%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- STRUCTURAL MODEL
%
%=========================================================================
% FUNCTION:         Estimates different versions of the structural model 
%                   across bootstrap samples and specifications.
% DATA:             
%
% INPUT:            takes input from Stata processed files and from 
%                   
% EXTERNAL CALLS:   fminunc 
%                   ComputeMu.m
%                   FunctionMig.m 
%
% NOTE:             This code implements two stage estimation of the 
%                   model. 
%                   First, we estimate base line shares (mmub and mmuc). 
%                   Next, the code contains an outer loop to generate 
%                   bootstrap samples. 
%                   Next, the code contains inner loops across different 
%                   model specifications and estimate alpha and delta.
%---------+---------+---------+---------+---------+---------+---------+---
% BUG REPORTS TO:   bart.bronnenberg@uvt.nl
%                   jdube@chicagobooth.edu
%                   gentzkow@chicagobooth.edu
%---------+---------+---------+---------+---------+---------+---------+---

clear all 

addpath('../external/');

global AGE YEARS XBMU XDMU Q MODIFIER WEIGHT ;

%---------+---------+---------+---------+---------+---------+---------+---
%         constants 
%---------+---------+---------+---------+---------+---------+---------+---

Get_Constants ;

%---------+---------+---------+---------+---------+---------+---------+---
%         data (fields are listed in the first row of each file)
%---------+---------+---------+---------+---------+---------+---------+---

lhs = dlmread('..\external\lhs_mig.csv',',',1,0);
X_demo = dlmread('..\external\X_demo_mig.csv',',',1,0);
X_stb = dlmread('..\external\X_stb_mig.csv',',',1,0);
X_stc = dlmread('..\external\X_stc_mig.csv',',',1,0);
hh_char = dlmread('..\temp\hh_char.csv',',',1,0);
mod_char = dlmread('..\temp\mod_char.csv',',',1,0);
lhs_screen = dlmread('..\external\lhs_mig_screen.csv',',',1,0);

%check that order is the same 
same_order = [X_demo(:,1)==X_stc(:,1)& ...
                        X_demo(:,1)==X_stb(:,1)] ; 
                    
if 1-isempty(find(same_order==0)) ,
    error('HH not in the same order') 
end

if sum(sum(lhs(:,1:2)==lhs_screen(:,1:2)))~=2*length(lhs) , 
    error('inconsistent file organization ')
end

hh_char(:,3) = min(hh_char(:,2:3),[],2) ; %cap years at age

%---------+---------+---------+---------+---------+---------+---------+---
%         mmuc & mmub: different definitions of mu_current and mu_born;  
%         order determined in the switch statement.
%---------+---------+---------+---------+---------+---------+---------+---

hh = lhs(:,1) ;               %household index
mod = lhs(:,2) ;              %module index
N = size(lhs,1) ;             %number of observations

for ip = 1:4 ,
    switch ip ,
        case 1, lab = 'purch' ;
        case 2, lab = 'equiv' ;
        case 3, lab = 'exp' ;
        case 4, lab = 'unit' ;
    end
    b_demo = dlmread(strcat('..\external\b_demo_',lab,'.csv'),',',1,0);
    b_st = dlmread(strcat('..\external\b_st_',lab,'.csv'),',',1,0);
    [tmub tmuc] = computemu([hh mod],b_demo,b_st,X_demo,X_stb,X_stc);
    mmub(:,ip) = tmub ;
    mmuc(:,ip) = tmuc ; 
end ;

%---------+---------+---------+---------+---------+---------+---------+---
%         modshare: Module shares for weights. Store in same order as lhs. 
%---------+---------+---------+---------+---------+---------+---------+---

umod = unique(b_demo(:,1)) ;  %unique modules

for mm = 1:length(umod) ,
    index = find(lhs(:,2)==umod(mm)) ; 
    ms = mean(lhs(index,3)./(sum(lhs(index,3:4),2))) ;
    modshare(index,1) = ms ;
end

%---------+---------+---------+---------+---------+---------+---------+---
%         moddesc: sources of heterogeneity  
%---------+---------+---------+---------+---------+---------+---------+---

[~,hindex] = ismember(hh,hh_char(:,1));
[~,mindex] = ismember(mod,mod_char(:,1));

age = hh_char(hindex,2);
years = hh_char(hindex,3);
moddesc(:,1:2) = mod_char(mindex,[2 3]) ; %advertising, social visibility
moddesc(:,3) = lhs_screen(:,3) ; %both available
for vv = 1:4 , % share, frequency, spatial var1, spatial var2
    unilist = unique(lhs_screen(:,[2 vv+3]),'rows') ;
    med = median(unilist(:,2)) ;
    moddesc(:,vv+3) = lhs_screen(:,vv+3)>med ;
end

%---------+---------+---------+---------+---------+---------+---------+---
%         moddesc_all: sources of heterogeneity using all consumers 
%                      regardless of gap  
%---------+---------+---------+---------+---------+---------+---------+---

b_demo = dlmread(strcat('..\external\b_demo_purch.csv'),',',1,0);
b_st = dlmread(strcat('..\external\b_st_purch.csv'),',',1,0);
lhs_all = dlmread('..\external\lhs_mig_all.csv',',',1,0);
X_demo_all = dlmread('..\external\X_demo_mig_all.csv',',',1,0);
X_stb_all = dlmread('..\external\X_stb_mig_all.csv',',',1,0);
X_stc_all = dlmread('..\external\X_stc_mig_all.csv',',',1,0);
N_all = size(lhs_all,1) ;             %number of observations
hh_all = lhs_all(:,1) ;               %household index
mod_all = lhs_all(:,2) ;              %module index
[mub_all muc_all] = ...
    computemu([hh_all mod_all],b_demo,b_st,X_demo_all,X_stb_all,X_stc_all);
[~,hindex] = ismember(hh_all,hh_char(:,1));
[~,mindex] = ismember(mod_all,mod_char(:,1));
age_all = hh_char(hindex,2);
years_all = hh_char(hindex,3);
purch_all = lhs_all(:,4:5) ;
gap_all = lhs_all(:,3) ;
for vv = 1:2 ,
    moddesc_all(:,vv) = gap_all>(0 + (vv==2)*5) ;
end

%---------+---------+---------+---------+---------+---------+---------+---
%         initialize estimation results
%---------+---------+---------+---------+---------+---------+---------+---

par_table6 = NaN(npar,R) ;   %alpha and delta estimates
par_table8 = NaN(npar,R,2) ;   
par_appendix1 = NaN(npar,R,4) ;
par_robust2 = NaN(npar,R,7) ;
par_generations3 = NaN(npar+1,R,7) ;
keepavmu = NaN([R,1]);       %average share among life timers
keepfval = NaN([R,1]);       

%---------+---------+---------+---------+---------+---------+---------+---
%         Outer loop to generate bootstrap samples across replications
%         The index ibs (IndexBootStrap) is generated to extract data
%---------+---------+---------+---------+---------+---------+---------+---

for rep = 1:R ;
    ibs = [] ; ibs_all = [] ;
    if rep>1 ,
        index = umod(keepmodind(:,rep)) ; %generate bootstrap index
        for ik = 1:length(index) , 
            partind = find(lhs(:,2)==index(ik));
            ibs = [ibs ; partind ] ; %data index  
            partind_all = find(lhs_all(:,2)==index(ik));
            ibs_all = [ibs_all ; partind_all ] ; %data index for all gaps
        end ; 
    else   
        ibs = [1:N]' ; ibs_all = [1:N_all]' ; %regular sample
    end ;
    N = length(ibs) ; N_all = length(ibs_all) ; 

%---------+---------+---------+---------+---------+---------+---------+---
%         Extract necessary data for each bootstrap sample
%---------+---------+---------+---------+---------+---------+---------+---
    purch = lhs(:,3:4) ; 

    TOTPURCH = sum(purch(ibs,:),2) ; 
    Q = purch(ibs,1)./TOTPURCH ;
    MODSHARE = modshare(ibs,1) ; 
    XBMU = mmub(ibs,1) ; 
    XDMU = mmuc(ibs,1)-XBMU ; 
    AGE = age(ibs) ; YEARS = years(ibs) ;
    
%---------+---------+---------+---------+---------+---------+---------+---
%         Inner loop to estimate alpha and delta across 
%         alternative specifications and robustness checks
%---------+---------+---------+---------+---------+---------+---------+---
    %base case, advertising, social vis., and robustness
    for sp = 1:8 , 
        par = NaN(npar,1) ;
        WEIGHT = 1 ;

        if sp == 1 , MODIFIER = zeros(N,1)  ;
        else MODIFIER = moddesc(ibs,sp-1) ; %dummy for data split
        end
        
        par0 = zeros(npar,1) ;
        par = fminunc(@FunctionMig, par0, options) ;
        [fval,db,avmu,p,funcval] = FunctionMig(par) ;
        par = exp(par)./(1+exp(par)) ;
        if sp==1 , par_table6(:,rep) = par ; 
            keepavmu(rep,1) = avmu; keepfval(rep,1) = funcval ; 
        elseif sp<4, par_table8(:,rep,sp-1) = par ;
        else par_robust2(:,rep,sp-3) = par ;
        end
        
        [rep sp]
    end
    
    %overlapping generations 
    mina = min(AGE) ; maxa = max(AGE) ;
    for sp = 1:7 ;
        par = NaN(npar,1) ;
        WEIGHT = 1 ;
        MODIFIER = zeros(N,1)  ;
        
        par0 = zeros(npar,1) ;
        AGE = age(ibs) + (sp-1)*10 ; %add treatment years in birth state 
        par = fminunc(@FunctionMig, par0, options) ;
        [fval, beta] = FunctionMig(par) ;
        par = exp(par)./(1+exp(par)) ;
        par_generations3(1:npar,rep,sp) = par ;
        par_generations3(end,rep,sp) = Get_AgeDist([mina:maxa])*...
                                       beta(1+(sp-1)*10,mina-1:maxa-1)' ;
                
        [rep sp+8]
    end
    AGE = age(ibs) ;
    
    %alternative dependent variables
    for sp = 1:4 ,
        par = NaN(npar,1) ;        
        WEIGHT=1  ;
        MODIFIER = zeros(N,1)  ;
        TOTPURCH = sum(lhs(ibs,1+2*sp:2+2*sp),2) ;
        Q = lhs(ibs,1+2*sp)./TOTPURCH ;
        XBMU = mmub(ibs,sp) ;  %mu_hat changes with dep. var 
        XDMU = mmuc(ibs,sp)-XBMU ;

        par0 = zeros(npar,1) ;
        par = fminunc(@FunctionMig, par0, options) ;  
        if sp == 1 ;
            [func, bat, avmu, q_hat] = FunctionMig(par) ;  
            select = isnan(Q.*XBMU.*XDMU)==0 ; %see flagged cases in computemu.m
            e2 = (Q(select)-q_hat(select)).^2 ;
            xtemp = [ones(N,1) (MODSHARE.*(1-MODSHARE))./TOTPURCH] ;
            partemp = ((xtemp(select,:)'*xtemp(select,:))\...
                                (xtemp(select,:)'*e2));   
            WEIGHT = 1./(xtemp*partemp);   
            par = fminunc(@FunctionMig, par0, options) ;  
        end    
        par = exp(par)./(1+exp(par)) ;
        par_appendix1(:,rep,sp) = par ;
        
        [rep sp+15]    
    end
    
    %include all consumers regardless of age gap
    TOTPURCH = sum(purch_all(ibs_all,:),2) ; 
    Q = purch_all(ibs_all,1)./TOTPURCH ;
    XBMU = mub_all(ibs_all) ;
    XDMU = muc_all(ibs_all)-XBMU ;
    AGE = age_all(ibs_all) ; YEARS = years_all(ibs_all) ;
    for sp = 1:2 ,    
        par = NaN(npar,1) ;
        WEIGHT=1  ;
        MODIFIER = moddesc_all(ibs_all,sp) ;
        par0 = zeros(npar,1) ;
        par = fminunc(@FunctionMig, par0, options) ;      
        par = exp(par)./(1+exp(par)) ;
        par_robust2(:,rep,sp+5) = par ;
 
        [rep sp+19]    
    end
    save '..\temp\results.mat' par_table6 par_table8 par_appendix1 ...
          par_robust2 par_generations3 keepavmu keepfval ;
end

% Save computed parameters to temp (needed in computation of figures and counterfactuals)
alpha  = par_table6(1,1) ; 
delta  = par_table6(2,1) ;
save '..\temp\alphadelta.mat' alpha delta ; %for counterfactuals


%save for plotting figures later on
mub = mmub(:,1) ; muc = mmuc(:,1) ;
purch = lhs(:,3:4) ;
save '..\temp\ToFigures.mat' years age mub muc purch modshare;

