/**********************************************************
 *
 * ShareChange.DO: Compute typical annual share movement
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
set mem 2G
set matsize 5000
set more off
set seed 04271975
set sortseed 04271975

use "..\external\hh_module_month.dta", clear

*collapse to year module  

gen date = dofm(ym)
gen y = year(date) 
gen m = month(date)
gen ym1 = y*100+m 
drop if ym1==200610
gen y1 = 0 
replace y1 = 1 if ym1 > 200710
gen totpurch = purch1+purch2 
collapse (sum) totpurch purch1, by(y1 module)

*compute absolute share difference year over year for 
*modules with more than 5000 purchases. 
gen share = purch1/totpurch
drop purch1
reshape wide share totpurch, i(module) j(y1)
gen totpurch = totpurch1+totpurch0
drop if totpurch <5000 
drop if totpurch == .
gen absdiv = abs(share1-share0)
gen large = absdiv>0.12
summarize absdiv large
sa "..\temp\text.dta", replace

cap log close