/**********************************************************
 *
 * STATA_TO_MATLAB.DO: Output delimited data file that
 *  will be used by Matlab scripts as input.
 *
 **********************************************************/

cap log close
cap log using "../output/format_data_bp.log", text replace
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version
version 10
clear all
set mem 800m
set matsize 5000
set more off
tempfile state_temp
set seed 4271975
set sortseed 4271975
adopath + ../external/

loadglob using input_param.txt

tempfile temp_hh_mig temp_hh_nonmig temp_mod

*****************************************************************
* TEMPORARY FILES
*****************************************************************
insheet using ../output/sample_hh_mig.csv, clear
save `temp_hh_mig', replace
insheet using ../output/sample_hh_nonmig.csv, clear
save `temp_hh_nonmig', replace
insheet using ../output/sample_mod.csv, clear
save `temp_mod', replace

*****************************************************************
* OUTPUT LHS VARIABLE
*****************************************************************
u hhld_id module pairid $q1 $q2 launch_year1 umbrella_year1 launch_year2 umbrella_year2 using ../external/by_hh_pair.dta, clear

* restrict sample of modules
mmerge module using `temp_mod', type(n:1) unmatched(master)
drop if _merge==1
drop if $q1==0 & $q2==0

* restrict sample of hhs
mmerge hhld_id using `temp_hh_nonmig', type(n:1) unmatched(master) _merge(_nonmig)
mmerge hhld_id using `temp_hh_mig', type(n:1) unmatched(master) _merge(_mig)
drop if _nonmig==1 & _mig==1
gen nonmig = _nonmig==3
bysort pairid: egen nobs = total(nonmig)
drop if nobs<500 
drop nobs

sort hhld_id module pairid
outsheet hhld_id module pairid $q1 $q2 launch_year1 umbrella_year1 launch_year2 umbrella_year2 using ../output/lhs_nonmig_bp.csv if nonmig==1, comma names replace
outsheet hhld_id module pairid $q1 $q2 launch_year1 umbrella_year1 launch_year2 umbrella_year2 using ../output/lhs_mig_bp.csv if nonmig==0, comma names replace
