%=========================================================================
%
%                   ENDOGENOUS PREFERENCES -- FIGURES.m
%
%=========================================================================
% FUNCTION:         This code creates 3D graphs and checkerboards of beta.
% DATA:             Estimated values for selected parameters plus 
%                   [years age mub muc purch] from RunStrucModel
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
%         3D graph of beta as a function of age and years         
%---------+---------+---------+---------+---------+---------+---------+---

clear all
load ..\temp\ToFigures.mat ;
k = isnan(mub.*muc)==0 ; %see flagged cases in computemu.m
years = years(k) ; age = age(k) ; mub =mub(k) ; muc=muc(k) ;
purch=purch(k) ; 

load ..\temp\alphadelta ; 
d = delta.^[0:200]' ;
dd = cumsum(d) ;

%in increments of 'step' years
step = 5 ;
top = round(80/step) ; 
xlab = [0:step:top*step] ; %labels
tix = 2:2:top ;
tixlab = 2*step:2*step:top*step ;

%age and years to create index
agemove = age-years ; 
dam = 1+floor(agemove/step) ; %age at move
dsm = 1+floor(years/step) ;   %years since move

%empirical definition of beta
purchshare = purch(:,1)./(sum(purch,2)+eps) ;
beta = (purchshare-mub)./(muc-mub+eps) ;
weight = (muc-mub+eps).^2;

%compute averages and predictions
clear m v n; %mean beta, predicted beta, number of obs with [age, years]
for i=1:top ,
    for j=1:top , 
        index = find(dam==i&dsm==j) ;
        nobs = length(index) ;
        v(i,j) = Get_Beta_at(i*step,j*step, alpha, dd) ; 
        n(i,j) = nobs ;
        if i+j>top+1 | nobs<750  , % plot only with sufficient obs. 
            m(i,j) = NaN ;
        else
            m(i,j) = sum(beta(index).*weight(index))/sum(weight(index)) ; 
        end        
    end
end
which = find(isnan(m)) ;
v(which) = NaN ; %use m as a mask to blank out v

%plot 
colormap(gray(100))
for k = 1:2 ,
    if k==1 ,
        plotvar = m ;
        ti = '\beta_{ij} - average'
    elseif k==2
        plotvar = v ;
        ti = '\beta (a,t)'
    end
    
    %3d bar plot
    bar3(plotvar,1,'w'), ylabel('age at move'), xlabel('years since move')
    set(gca,'XTick',tix) ; set(gca,'XTickLabel',tixlab) ;
    set(gca,'YTick',tix) ; set(gca,'YTickLabel',tixlab) ;
    axis([0 top+.5 0 top+.5 0 1])
    title(char(ti)) ;
    fn = strcat('..\output\figures',ti,'bars.eps') ;
    print('-depsc', '-r300', '-painters', fn) ;
    
    %checker plot
    nr = size(plotvar,1) ;
    pcolor(plotvar(nr:-1:1,:)) ;
    ylabel('age at move') ;
    xlabel('years since move') ;  
    set(gca,'XTick',tix) ; set(gca,'XTickLabel',tixlab) ;
    set(gca,'YTick',tix) ; set(gca,'YTickLabel',[top*step:-2*step:2*step]) ;
    colorbar ; %axis([0 80 0 80 .6 1]) ;
    fn = strcat('..\output\figures',ti,'checker.eps') ;
    print('-depsc', '-r300', '-painters', fn) ;
end

%residual graph
plotvar = m - v ;
nr = size(plotvar,1) ;
ti = 'average \beta_{ij}-\beta(a,t) '

%3d plot
bar3(plotvar,1,'w'), ylabel('age at move'), xlabel('years since move')
set(gca,'XTick',tix) ; set(gca,'XTickLabel',tixlab) ;
set(gca,'YTick',tix) ; set(gca,'YTickLabel',tixlab) ;
axis([0 top+.5 0 top+.5 -.2 .2])
title(char(ti)) ;
fn = strcat('..\output\figures\residbars.eps') ;
print('-depsc', '-r300', '-painters', fn) ;

%checker plot
pcolor(plotvar(nr:-1:1,:)) ;
xlabel('years since move') ;  ylabel('age at move') ;
set(gca,'XTick',tix) ; set(gca,'XTickLabel',tixlab) ;
set(gca,'YTick',tix) ; set(gca,'YTickLabel',[top*step:-2*step:2*step]) ;
colorbar ; %axis([0 80 0 80 .6 1]) ;
fn = strcat('..\output\figures\residchecker.eps') ;
print('-depsc', '-r300', '-painters', fn) ;


