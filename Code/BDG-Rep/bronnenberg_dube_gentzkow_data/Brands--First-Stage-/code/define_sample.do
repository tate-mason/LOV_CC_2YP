/**********************************************************
 *
 * DEFINE_SAMPLE.DO
 *
 **********************************************************/

cap log close
log using ..\output\define_sample.log, text replace
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version
version 11
clear all
set mem 500m
set matsize 5000
set more off
set seed 04271975
set sortseed 04271975
adopath + ..\external
loadglob using input_param.txt

tempfile temp temp_top2same

*****************************************************************
* DEFINE SAMPLE OF MODULES
*****************************************************************
insheet using ../external/module_top2same.csv, clear
keep if top2same==1
save module using `temp_top2same', replace

u module hh_all using ../external/by_module.dta, clear
drop if hh_all<$hhcutoff
mmerge module using `temp_top2same', type(1:1) unmatched(master) ukeep()
drop if _merge==3

sort module
outsheet module using ../output/sample_mod.csv, comma replace names

save module using `temp', replace

*****************************************************************
* STORE LIST OF HHS W/ PURCHASES IN SELECTED MODULES
*****************************************************************
u module hhld_id $q1 $q2 using ../external/by_hh_module.dta, clear
mmerge module using `temp', type(n:1) unmatched(master)
drop if _merge==1
drop if $q1==0 & $q2==0
keep hhld_id
duplicates drop
save `temp', replace

*****************************************************************
* DEFINE SAMPLE OF HOUSEHOLDS
*****************************************************************
u hhld_id age gap keeper state_born state_curr if keeper==1 using ..\external\survey.dta, clear

* drop cases where age, years, and age_moved were mutually inconsistent
drop if gap==.

* drop cases where age out of range
drop if age>$maxage | age<$minage

* drop the one household in the data whose current state is reported as Hawaii or Alaska
drop if state_curr=="HI" | state_curr=="AK" 

* drop households born in Hawaii or Alaska
drop if state_born=="HI" | state_born=="AK"

* drop cases for which we have no purchase data
mmerge hhld_id using `temp', type(n:1) unmatched(master)
drop if _merge==1

sort hhld_id
outsheet hhld_id gap using ../output/sample_hh_mig_all.csv if state_born~=state_curr, comma replace names

* drop cases where gap out of range 
* gap is defined to be  0 for life time residents in ..\derived\Homescan (Survey)\build.do 

drop if gap>$maxgap

sort hhld_id
outsheet hhld_id using ../output/sample_hh_mig.csv if state_born~=state_curr, comma replace names
outsheet hhld_id using ../output/sample_hh_nonmig.csv if state_born==state_curr, comma replace names

cap log close
