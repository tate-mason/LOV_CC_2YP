/**********************************************************
 *
 * FIGURES.DO
 *
 **********************************************************/

cap log close
log using ..\output\figures.log, text replace
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version
version 10
clear all
set mem 500m
set matsize 5000
set more off
set seed 04271975
set sortseed 04271975
adopath + ..\external
loadglob using ..\temp\figures.txt
set scheme s1mono

*****************************************************************
* FIGURE <fig:SummaryStats>
*****************************************************************
* Age
u ..\temp\hh_mig, clear
histogram age, fraction start(5) width(10) gap(10) title("Age", size(medium)) xlabel(0(20)80, noticks) xtitle(Years) ///
	yscale(range(0 .31)) ylabel(0(0.1).3) ytitle("Share") name(age, replace)

* Age leaving state of birth
histogram age_move, fraction start(-5) width(10) gap(10) title("Age Leaving State of Birth", size(medium)) xlabel(0(20)80, noticks) xtitle(Years) ///
	yscale(range(0 .31)) ylabel(0(0.1).3) ytitle("Share") name(age_move, replace)

* Years living in current state
histogram years, fraction start(-5) width(10) gap(10) title("Years Living in Current State", size(medium)) xlabel(0(20)80, noticks) xtitle(Years) ///
	yscale(range(0 .21)) ylabel(0(0.1).2) ytitle("Share") name(years, replace)

* Gap
histogram gap, fraction start(-.5) width(1) gap(10) title("Gap", size(medium)) xscale(range(0 5)) xlabel(0(1)5, noticks) xtitle(Years) ///
	yscale(range(0 .84)) ylabel(0(.2).8) ytitle("Share") name(gap, replace)

* Combine
graph combine age age_move years gap
graph export ..\output\figures\summarystats.eps, as(eps) replace
graph drop _all


*****************************************************************
* FIGURE <fig:Years>
*****************************************************************
u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years_cat)

local xlab "0-4 5-9 10-14 15-19 20-24 25-29 30-34 35-39 40-44 45-49 50-54 55-59 60+" 
local options "ylabel(0(.1)1) yline(0 1, lpattern(dash)) yscale(range(-.1 1.1)) ytitle(Relative Share ({&beta} {subscript:ij})) xtitle(Years Since Move) label(`xlab')"
quietly tab years_cat, gen(zyears)

gen w1nd = totpurch*sqdifnd/(modshare*(1-modshare))
gen w1 = totpurch*sqdif/(modshare*(1-modshare))

reg betahatnd zyears1-zyears13 [aw=sqdifnd], cluster(module) nocons
plotcoeffs zyears1-zyears13, `options'
graph export ..\output\figures\years_none.eps, as(eps) replace

reg betahat zyears1-zyears13 [aw=sqdif], cluster(module) nocons
plotcoeffs zyears1-zyears13, `options'
graph export ..\output\figures\years_demo.eps, as(eps) replace

*****************************************************************
* FIGURE <fig:AgeMove>
*****************************************************************
u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(age_move_cat)

local xlab "0-4 5-9 10-14 15-19 20-24 25-29 30-34 35-39 40-44 45-49 50-54 55-59 60+" 
local options "ylabel(0(.1)1) yline(0 1, lpattern(dash)) yscale(range(-.1 1.1)) ytitle(Relative Share ({&beta} {subscript:ij})) xtitle(Age at Move) label(`xlab')"
quietly tab age_move_cat, gen(zagemove)

gen w1nd = totpurch*sqdifnd/(modshare*(1-modshare))
gen w1 = totpurch*sqdif/(modshare*(1-modshare))

reg betahatnd zagemove1-zagemove13 [aw=sqdifnd], cluster(module) nocons
plotcoeffs zagemove1-zagemove13 , `options'
graph export ..\output\figures\agemove_none.eps, as(eps) replace

reg betahat zagemove1-zagemove13 [aw=sqdif], cluster(module) nocons
plotcoeffs zagemove1-zagemove13 , `options'
graph export ..\output\figures\agemove_demo.eps, as(eps) replace

*****************************************************************
* FIGURE <fig:AgeMove_recent>
*****************************************************************
u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years age_move_cat)

local xlab "20-24 25-29 30-34 35-39 40-44 45-49 50-54 55-59 60+" 
local options "ylabel(0(.1)1) yline(0 1, lpattern(dash)) yscale(range(-.1 1.1)) ytitle(Relative Share ({&beta} {subscript:ij})) xtitle(Age at Move) label(`xlab')"
quietly tab age_move_cat, gen(zagemove)

gen w1nd = totpurch*sqdifnd/(modshare*(1-modshare))
gen w1 = totpurch*sqdif/(modshare*(1-modshare))

reg betahatnd zagemove4-zagemove13 if years<=5 [aw=sqdifnd], cluster(module) nocons
plotcoeffs zagemove4-zagemove13 , `options'
graph export ..\output\figures\agemove_recent_none.eps, as(eps) replace

reg betahat zagemove4-zagemove13 if years<=5  [aw=sqdif], cluster(module) nocons
plotcoeffs zagemove4-zagemove13 , `options'
graph export ..\output\figures\agemove_recent_demo.eps, as(eps) replace

*****************************************************************
* FIGURE <fig:Years_25plus>
*****************************************************************
u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years age_move years_cat)

local xlab "0-4 5-9 10-14 15-19 20-24 25-29 30-34 35-39 40-44 45-49 50-54" 
local options "ylabel(0(.1)1) yline(0 1, lpattern(dash)) yscale(range(-.1 1.1)) ytitle(Relative Share ({&beta} {subscript:ij})) xtitle(Years Since Move) label(`xlab')"
quietly tab years_cat, gen(zyears)

gen w1nd = totpurch*sqdifnd/(modshare*(1-modshare))
gen w1 = totpurch*sqdif/(modshare*(1-modshare))

reg betahatnd zyears1-zyears11 if age_move>25 & years_cat<=50 [aw=sqdifnd], cluster(module) nocons
plotcoeffs zyears1-zyears11, `options'
graph export ..\output\figures\years25_none.eps, as(eps) replace

reg betahat zyears1-zyears11 if age_move>25 & years_cat<=50 [aw=sqdif], cluster(module) nocons
plotcoeffs zyears1-zyears11, `options'
graph export ..\output\figures\years25_demo.eps, as(eps) replace

*****************************************************************
* FIGURE <fig:RecentMovers>
*****************************************************************
u ..\temp\purch_mig_month, clear

gen year = year(dofm(ym))
gen month = month(dofm(ym))

keep if (years==1 | years==0) & gap==0

* note: survey fielded 9/13/08 - 10/1/08
drop if year==2008 & month>=10

quietly tab ym, gen(zym)
egen hhmod=group(hhld_id module)
local xlab "10/06 11/06 12/06 1/07 2/07 3/07 4/07 5/07 6/07 7/07 8/07 9/07 10/07 11/07 12/07 1/08 2/08 3/08 4/08 5/08 6/08 7/08 8/08 9/08 10/08"
local options "ylabel(0(.1)1) yline($alpha, lpattern(dash) lcolor(red)) yline(0 1, lpattern(dash)) yscale(range(-.1 1.1)) ytitle(Relative Share ({&beta} {subscript:ij})) xtitle(Month/Year) label(`xlab')"

gen w1nd = totpurch*sqdifnd/(modshare*(1-modshare))

* <1 Year
reg betahatnd zym1-zym24 if years==0 [aw=sqdifnd], cluster(module) nocons
plotcoeffs zym1-zym24, `options' note("years living in current state: less than 1")
graph export ..\output\figures\years0_none.eps, as(eps) replace

* 1-2 Years
reg betahatnd zym1-zym24 if years==1 [aw=sqdifnd], cluster(module) nocons
plotcoeffs zym1-zym24, `options' note("Years Living in Current State: 1")
graph export ..\output\figures\years1_none.eps, as(eps) replace

cap log close
