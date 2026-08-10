/**********************************************************
 *
 * STATA_TO_MATLAB.DO: Output delimited data file that
 *  will be used by Matlab scripts as input.
 *
 **********************************************************/

cap log close
cap log using "../output/format_data.log", text replace
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version
version 11
clear all
set mem 800m
set matsize 5000
set more off
tempfile state_temp
set seed 4271975
set sortseed 4271975
adopath + ../external/

loadglob using input_param.txt

tempfile temp_hh_mig temp_hh_mig_all temp_hh_nonmig temp_mod temp_sales temp_sales2 temp_both module_sd

*****************************************************************
* TEMPORARY FILES
*****************************************************************
insheet using ../output/sample_hh_mig.csv, clear
save `temp_hh_mig', replace
insheet using ../output/sample_hh_mig_all.csv, clear
save `temp_hh_mig_all', replace
insheet using ../output/sample_hh_nonmig.csv, clear
save `temp_hh_nonmig', replace
insheet using ../output/sample_mod.csv, clear
save `temp_mod', replace

*****************************************************************
* OUTPUT LHS VARIABLE
*****************************************************************
u hhld_id module $q1 $q2 $q3 $q4 $q5 $q6 $q7 $q8 purch_gen purch_oth using ../external/by_hh_module.dta, clear

* restrict sample of modules
mmerge module using `temp_mod', type(n:1) unmatched(master)
drop if _merge==1
drop if $q1==0 & $q2==0

* restrict sample of hhs
mmerge hhld_id using `temp_hh_nonmig', type(n:1) unmatched(master) _merge(_nonmig)
mmerge hhld_id using `temp_hh_mig', type(n:1) unmatched(master) _merge(_mig)
mmerge hhld_id using `temp_hh_mig_all', type(n:1) unmatched(master) _merge(_mig_all)
drop if _nonmig==1 & _mig==1 & _mig_all==1
gen nonmig = _nonmig==3
gen mig = _mig==3 
gen mig_all = _mig_all==3 

save `temp_sales'
sort hhld_id module
outsheet hhld_id module $q1 $q2 $q3 $q4 $q5 $q6 $q7 $q8 using ../output/lhs_nonmig.csv if nonmig==1, comma names replace
outsheet hhld_id module $q1 $q2 $q3 $q4 $q5 $q6 $q7 $q8 using ../output/lhs_mig.csv if mig==1, comma names replace
outsheet hhld_id module gap $q1 $q2 $q3 $q4 $q5 $q6 $q7 $q8 using ../output/lhs_mig_all.csv if mig_all==1, comma names replace

*****************************************************************
* OUTPUT RHS VARIABLES
*****************************************************************
u hhld_id state_curr state_born age years age_move linc hs col grad emp keeper using ..\external\survey.dta if keeper==1, clear
mmerge hhld_id using ../external/demos, type(n:1) unmatched(master) ukeep(hispseg hispanic race rentown)

* restrict sample of hhs
mmerge hhld_id using `temp_hh_nonmig', type(n:1) unmatched(master) _merge(_nonmig)
mmerge hhld_id using `temp_hh_mig', type(n:1) unmatched(master) _merge(_mig)
mmerge hhld_id using `temp_hh_mig_all', type(n:1) unmatched(master) _merge(_mig_all)
drop if _nonmig==1 & _mig==1 & _mig_all==1
gen nonmig = _nonmig==3
gen mig = _mig==3 
gen mig_all = _mig_all==3 

* confirm that the sets of birth and current states are the same
levelsof state_born, clean local(stblist)
levelsof state_curr, clean local(stclist)
assert "`stblist'"=="`stclist'"

* create dummies for states, being careful to use same encoding for state_born & state_curr
local i = 1
foreach S in `stclist' {
	gen zstb`i' = state_born=="`S'"
	gen zstc`i' = state_curr=="`S'"
	local i = `i'+1
}

* create dummies for demos
gen age_cat = min(max(floor(age/5)*5,30),65)
foreach V in age_cat hispanic race emp {
	quietly tab `V', gen(z`V')
}

sort hhld_id
outsheet hhld_id $demos using ../output/X_demo_nonmig.csv if nonmig==1, comma names replace
outsheet hhld_id $demos using ../output/X_demo_mig.csv if mig==1, comma names replace
outsheet hhld_id $demos using ../output/X_demo_mig_all.csv if mig_all==1, comma names replace
outsheet hhld_id zstc* using ../output/X_st_nonmig.csv if nonmig==1, comma names replace
outsheet hhld_id zstc* using ../output/X_stc_mig.csv if mig==1, comma names replace
outsheet hhld_id zstc* using ../output/X_stc_mig_all.csv if mig_all==1, comma names replace
outsheet hhld_id zstb* using ../output/X_stb_mig.csv if mig==1, comma names replace
outsheet hhld_id zstb* using ../output/X_stb_mig_all.csv if mig_all==1, comma names replace


*****************************************************************
* FLAG STATE-MODULES WITH RECORDED SALES FOR BOTH TOP BRANDS
* This is code added for robustness checks. Create an index 
* that maps into lhs_mig which gives a 1 when we record sales
* for both top brands (based on both migrants and non-migrants)
* Primary keys are:
*     hhld_id-module for 'both' 
*     module for 'topshare' and 'totpurch' (=purchase frequency)
*****************************************************************

u hhld_id state_curr state_born keeper using ..\external\survey.dta if keeper==1, clear
mmerge hhld_id using `temp_sales', type(1:n) unmatched(none)
sort state_curr state_born module
gen totpurch = $q1+$q2+purch_gen+purch_oth 
save `temp_sales2', replace

sort module state_curr 
collapse (sum) $q1 $q2, by(module state_curr) 
gen pair_available = $q1*$q2>0 
gen state_field = state_curr 
drop state_curr  
save `temp_both', replace 

use `temp_sales2', clear 
keep if nonmig==1 
gen share = $q1/($q1+$q2)
collapse (mean) $q1 $q2 share, by(module state_curr) 
gen avshare = $q1/($q1+$q2)
collapse (sd) geog_sd = avshare share_sd = share, by(module) 
keep module geog_sd share_sd 
save `module_sd'

use `temp_sales2', clear 
mmerge state_curr module using `temp_both', type(n:1) umatch(state_field module) urename("pair_available pav_curr") 
mmerge state_born module using `temp_both', type(n:1) umatch(state_field module) urename("pair_available pav_born") 
mmerge module using `module_sd', type(n:1) 
gen both = pav_born*pav_curr 
replace both = 0 if both == . 
keep hhld_id module both geog_sd share_sd
save `temp_both', replace

use `temp_sales2', clear
gen toppurch = $q1+$q2 
collapse (sum) toppurch totpurch, by(module)
gen topshare = toppurch/totpurch 
mmerge module using `temp_both', type(1:n) unmatched(none) 
drop if hhld_id == . 
keep hhld_id module both topshare totpurch geog_sd share_sd

mmerge hhld_id module using `temp_sales', type(1:1) unmatched(using) 
drop if _merge == 1 
save `temp_both', replace 

sort hhld_id module 

outsheet hhld_id module both topshare totpurch geog_sd share_sd using ../output/lhs_mig_screen.csv if mig==1, comma names replace 
