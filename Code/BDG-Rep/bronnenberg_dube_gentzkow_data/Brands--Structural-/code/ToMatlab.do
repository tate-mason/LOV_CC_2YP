/**********************************************************
 *
 * ToMatlab.DO
 *
 **********************************************************/

cap log close
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version
version 11
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
local survey ..\external\survey.dta
local modulechars ..\external\module_chars.csv
local hh_sample ..\external\sample_hh_mig.csv
local hh_sample_all ..\external\sample_hh_mig_all.csv
local lhs_mig ..\external\lhs_mig.csv
local lhs_nonmig ..\external\lhs_nonmig.csv
local addata ..\external\module_top2brands_dollars_2008.dta 
local visible ..\external\module_visibility.csv

*****************************************************************
* TEMP FILE WITH VARIABLES FROM SURVEY
*****************************************************************
tempfile hhs

insheet using `hh_sample', clear
save `hhs' , replace

u hhld_id age years keeper if keeper == 1 using `survey', clear
drop keeper

outsheet hhld_id age years using ../temp/hh_char.csv, comma names replace

*****************************************************************
* CREATE MODULE CHARACTERISTICS FILE
*****************************************************************
tempfile char
insheet using `modulechars', clear
foreach V in ad social nokids {
	egen `V'tot = rsum(`V'_np `V'_pd `V'_yl)
}
gen ad = adtot==3
gen social = socialtot==3
gen nokids = nokidstot>0
keep module ad social nokids
save `char', replace

tempfile mig 
insheet using `lhs_mig' , clear
save `mig', replace 
insheet using `lhs_nonmig', clear 
append using `mig' 
gen purch = purch1+purch2 
drop purch1 purch2
collapse (sum) purch, by(module)
egen medpurch = median(purch) 
gen big = purch>medpurch
keep module big 

tempfile char2 
mmerge module using `char', type(n:1) unmatched(none)
save `char2', replace

tempfile ads2 
use `addata', clear 
rename totaldols000 addols000
keep module addols000 
mmerge module using `char2', type(n:1) unmatched(using)
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
mmerge module using `ads2', type(n:1) unmatched(using)
save `vis1' 

correlate ad ad2 ad3 ad4 ad5 vis_module vis2 vis3 social


outsheet module ad4 vis2 nokids big using ../temp/mod_char.csv, comma names replace

cap log close
