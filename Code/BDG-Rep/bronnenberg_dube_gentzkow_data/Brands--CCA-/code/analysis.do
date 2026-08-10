/**********************************************************
 *
 * ANALYSIS.DO
 *
 **********************************************************/

cap log close
log using ..\output\analysis.log, text replace
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version
version 11
clear all
set mem 1g
set matsize 5000
set more off
set scheme s1mono

adopath + ..\external

local cca_hs_brand "..\temp\cca_hs_brand.dta"
local by_brand "..\external\by_brand.dta"
local by_module "..\external\by_module.dta"

tempfile temp

**********************************************************
* DESCRIBE MATCHES
**********************************************************
u `cca_hs_brand', clear	
keep brand module
duplicates drop
mmerge brand using `by_brand', type(n:1) unm(master) ukeep(rank)

egen numbr = sum(1), by(module)
egen numtop2 = sum(rank==1 | rank==2), by(module)
rankunique -rank, by(module) gen(rankindata)
gen perfmatch = numtop2==2

unique module if numtop2==0
unique module if numtop2==1
unique module if numtop2==2

save `temp'

**********************************************************
* SET UP DATA FOR PURCHASE SHARES
**********************************************************
u `cca_hs_brand', clear
mmerge brand using `temp', type(n:1) unm(master) ukeep(perfmatch rankindata numtop2 numbr)

keep if numbr>=2
keep if rankindata==1 | rankindata==2

mmerge brand using `by_brand', type(n:1) unm(master) ukeep(brand_name)
mmerge module using `by_module', type(n:1) unm(master) ukeep(module_name)

keep module brand module_name brand_name state share_hs share_cca perfmatch rankindata hhs_hs yrs_cca
reshape wide brand brand_name share_hs share_cca hhs_hs yrs_cca, i(module state) j(rankindata)
placevar state module module_name perfmatch
egen numstates = sum(1), by(module)

gen hhs_hs  = hhs_hs1 + hhs_hs2
gen yrs_cca = min(yrs_cca1, yrs_cca2)
drop hhs_hs1 hhs_hs2 yrs_cca1 yrs_cca2
gen ps_hs = share_hs1 / (share_hs1 + share_hs2)
gen ps_cca = share_cca1 / (share_cca1 + share_cca2)

save ../temp/rel_shares.dta, replace

*****************************************************************
* TABLE <Tab:CCA>
*****************************************************************
cap program drop addtotable
program addtotable
	local r2 = e(r2)
	local N = e(N)
	local mod = e(N_clust)
	local F = r(F)
	local p = r(p)
	matrix TABLE = (nullmat(TABLE) , (_b[ps_hs] \ _se[ps_hs] \ _b[_cons] \ _se[_cons] \ `p' \ `mod' \ `N'))
end

use ../temp/rel_shares.dta, clear


* Regressions
reg ps_cca ps_hs if perfmatch, cluster(module)
test (ps_hs=1) (_b[_cons]=0)

reg ps_cca ps_hs if perfmatch [aw=yrs_cca], cluster(module)
test (ps_hs=1) (_b[_cons]=0)
addtotable

reg ps_cca ps_hs if perfmatch & hhs_hs>200 [aw=yrs_cca], cluster(module)
test (ps_hs=1) (_b[_cons]=0)
addtotable

reg ps_cca ps_hs if perfmatch & hhs_hs>500 [aw=yrs_cca], cluster(module)
test (ps_hs=1) (_b[_cons]=0)
addtotable

matrix_to_txt, saving(..\output\tables.txt) mat(TABLE) format(%20.6f) title(<tab:CCA>) append

**********************************************************
* FIGURES
**********************************************************
use ../temp/rel_shares.dta, clear

* Histogram of shares
histogram ps_cca
graph export "..\output\figures\histogram_cca.eps", replace
histogram ps_hs
graph export "..\output\figures\histogram_hs.eps", replace


* Scatter Plot
scatter ps_cca ps_hs if perfmatch & hhs_hs>100 [aw=yrs_cca], msymbol(circle_hollow) msize(vsmall) xtitle(Purchase Share 2006-2008 (Homescan)) ytitle(Purchase Share 1948-1968 (CCA)) ///
	|| lfit ps_cca ps_hs if perfmatch & hhs_hs>100 [aw=yrs_cca],  legend(order(2))
graph export "..\output\figures\scatter.eps", replace

* Graphs of share_cca vs share_hs
replace module_name = subinstr(module_name,"/","-",.)
levelsof module_name if numstates>5, local(mod) 
foreach m in `mod' {
	levelsof brand_name1 if module_name=="`m'", local(br1) clean
	levelsof brand_name2 if module_name=="`m'", local(br2) clean
	scatter ps_cca ps_hs if module_name == "`m'", ///
		mlabel(state) title("`m'") subtitle(`"Brands: `br1' & `br2'"')  xtitle(Purchase Share 2006-2008 (Homescan)) ytitle(Purchase Share 1948-1968 (CCA))  ///
		xlabel(0(.1)1) ylabel(0(.1)1) scale(.8) yscale(range(0 1)) xscale(range(0 1))
	graph export "..\output\figures\predict_shares_`m'.eps", replace
} 

*****************************************************************
* OUTPUT STATE-MODULE PAIRS
*****************************************************************
use ../temp/rel_shares.dta, clear
keep if perfmatch
keep state module ps_cca ps_hs yrs_cca
outsheet using ../output/rel_shares.csv, comma names replace


cap log close