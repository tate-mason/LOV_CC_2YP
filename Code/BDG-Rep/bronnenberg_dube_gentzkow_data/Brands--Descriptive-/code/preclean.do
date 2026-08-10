/**********************************************************
 *
 * PRECLEAN.DO
 *
 **********************************************************/

cap log close
cap log using "../output/preclean.log", text replace
set linesize 255

*****************************************************************
* PRELIMINARIES
*****************************************************************
version 10
clear all
set mem 800m
set matsize 5000
set more off

tempfile temp_lhs temp_mu temp_samplehh temp_samplemod

*****************************************************************
* HH DATA (NON-MIGRANTs)
*****************************************************************
insheet using ../external/sample_hh_nonmig.csv, names comma clear
mmerge hhld_id using ../external/survey.dta, type(n:1) unm(master) uif(keeper==1) ukeep(age reg_born reg_curr state_born state_curr)
drop _merge
save ../temp/hh_nonmig.dta, replace

*****************************************************************
* HH DATA (MIGRANTS)
*****************************************************************
insheet using ../external/sample_hh_mig.csv, names comma clear
mmerge hhld_id using ../external/survey.dta, type(n:1) unm(master) uif(keeper==1) ukeep(years age age_move reg_born reg_curr state_born state_curr gap)
drop _merge

* define categorical years & age variables
gen years_cat = min(max(floor(years/5)*5,0),60) if years~=.
gen age_cat = min(max(floor(age/5)*5,30),65)
gen age_move_cat = min(max(floor(age_move/5)*5,0),60)

save ../temp/hh_mig.dta, replace

*****************************************************************
* PURCHASE DATA (NON-MIGRANTS)
*****************************************************************
insheet using ../external/lhs_nonmig.csv, names comma clear
gen share = purch1 / (purch1 + purch2)
save ../temp/purch_nonmig.dta, replace

*****************************************************************
* PURCHASE DATA (MIGRANTS)
*****************************************************************
insheet using ../external/lhs_mig.csv, names comma clear
save `temp_lhs', replace

insheet using ../temp/mu.csv, names comma clear
mmerge hhld_id module using `temp_lhs', type(1:1) unm(master) ukeep(purch1 purch2)
drop _merge

* recode mu
foreach V in mub muc mundb mundc {
	replace `V' = "" if `V' == "NaN"
}
destring mub muc mundb mundc, replace

* drop if no gap
drop if mundb==mundc 

* define share, betahat, sqdif, and variables need to compute weights 
gen totpurch = purch1 + purch2 
gen share = purch1 / totpurch
gen betahat = (share - mub) / (muc - mub)
gen sqdif = (muc - mub)^2
gen betahatnd = (share - mundb) / (mundc - mundb)
gen sqdifnd = (mundc - mundb)^2
by module, sort: egen modpurch = mean(totpurch) 
by module, sort: egen modshare = mean(share) 

save ../temp/purch_mig.dta, replace


*****************************************************************
* PURCHASE DATA (MIGRANTS BY MONTH)
*****************************************************************
insheet using ../temp/mu.csv, names comma clear
save `temp_mu', replace
insheet using ../external/sample_hh_mig.csv, names comma clear
save `temp_samplehh', replace
insheet using ../external/sample_mod.csv, names comma clear
save `temp_samplemod', replace

use hhld module ym purch1 purch2 using ../external/hh_module_month.dta, clear

* restrict to sample hhs & modules
mmerge hhld_id using `temp_samplehh', type(n:1) unm(none)
mmerge module using `temp_samplemod', type(n:1) unm(none)

* merge in other data
mmerge hhld_id module using `temp_mu', type(n:1) unm(master) ukeep(mu*)
mmerge hhld_id using ..\external\survey, type(n:1) unm(master) uif(keeper==1) ukeep(keeper years age_move gap)
drop _merge keeper

* recode mu
foreach V in mub muc mundb mundc {
	replace `V' = "" if `V' == "NaN"
}
destring mub muc mundb mundc, replace

* define share, betahat, and sqdif
gen totpurch = purch1 + purch2 
gen share = purch1 / totpurch
gen betahat = (share - mub) / (muc - mub)
gen sqdif = (muc - mub)^2
gen betahatnd = (share - mundb) / (mundc - mundb)
gen sqdifnd = (mundc - mundb)^2
by module, sort: egen modpurch = mean(totpurch)
by module, sort: egen modshare = mean(share) 
save ../temp/purch_mig_month.dta, replace


*****************************************************************
* PURCHASE DATA (MIGRANTS BRAND PAIRS)
*****************************************************************
tempfile temp_lhs2
insheet using ../external/lhs_mig_bp.csv, names comma clear
save `temp_lhs2', replace

insheet using ../temp/mu_bp.csv, names comma clear
mmerge hhld_id pairid using `temp_lhs2', type(1:1) unm(master) ukeep(module purch1 purch2 launch_year1 umbrella_year1 launch_year2 umbrella_year2)
drop _merge

* recode mu
foreach V in mub muc mundb mundc {
	replace `V' = "" if `V' == "NaN"
}
destring mub muc mundb mundc, replace

*drop if no gap
drop if mundb == mundc 

* define share, betahat, sqdif, and other stuff needed to compute weightsd
gen totpurch = purch1 + purch2 
gen share = purch1 / totpurch
gen betahat = (share - mub) / (muc - mub)
gen sqdif = (muc - mub)^2
gen betahatnd = (share - mundb) / (mundc - mundb)
gen sqdifnd = (mundc - mundb)^2
by pairid, sort: egen pairpurch = mean(totpurch) 
by pairid, sort: egen pairshare = mean(share)

* keep only biggest pair in each  module
egen max = max(pairpurch), by(module)
keep if pairpurch==max
drop max

save ../temp/purch_mig_bp.dta, replace

*****************************************************************
* CROSS-STATE STDEV OF PURCHASE SHARES
*****************************************************************
use hhld_id module share using ..\temp\purch_nonmig, clear
mmerge hhld_id using ..\temp\hh_nonmig, type(n:1) unm(master) ukeep(state_curr)
collapse (mean) stateshare = share, by(module state_curr)
collapse (sd) sdstateshare = stateshare, by(module)
save ../temp/stddev.dta, replace

cap log close
