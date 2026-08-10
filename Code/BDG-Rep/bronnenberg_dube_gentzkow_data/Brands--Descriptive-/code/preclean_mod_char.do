/**********************************************************
 *
 * ToMatlab.DO
 *
 **********************************************************/

cap log close
log using ..\output\preclean_mod_char.log, text replace
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version
version 10
clear
set mem 500m
set matsize 5000
set more off
set seed 04271975
set sortseed 04271975
adopath + ..\external
tempfile surtemp modtemp tertemp

**********************************************************
* FILES
**********************************************************
local addata ..\external\module_top2brands_dollars_2008.dta 
local visible ..\external\module_visibility.csv

*****************************************************************
* CREATE MODULE CHARACTERISTICS FILE
*****************************************************************

tempfile ads2 
use `addata', clear 
rename totaldols000 addols000
keep module addols000 
replace addols=0 if addols ==.
egen meddols = median(addols) 
gen ad2 = addols000>meddols 
egen ad25 = pctile(addols), p(25)
gen ad3 = addols>ad25 
egen ad75 = pctile(addols), p(75)
gen ad4 = addols>ad75 
gen ad5 = ad3+ad4 
save `ads2', replace 

tempfile vis1 
insheet using `visible', clear
rename visibility_module vis_module
gen vis2 = vis_module==2 
gen vis3 = vis_module>0 
keep module vis_module vis2 vis3
mmerge module using `ads2', type(n:1) unmatched(both)
save `vis1' 

keep module ad4 vis2
ren ad4 ad
ren vis2 social

replace ad=0 if ad==.

save ../temp/modchar.dta, replace

cap log close


