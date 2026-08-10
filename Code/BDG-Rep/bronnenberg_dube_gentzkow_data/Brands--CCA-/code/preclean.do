/**********************************************************
 *
 * PRECLEAN.DO  -- This file prepares the CCA and Homescan data for analysis
 *
 **********************************************************/

cap log close
log using ..\output\preclean.log, text replace
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version
version 10
clear all
set mem 2g
set matsize 5000
set more off

local hh_homescan "..\external\by_hh_brand.dta"
local cca "..\external\brand_shares.dta" 
local cross_br "..\external\CCA_Homescan_crosswalk.dta"
local demos "..\external\demos.dta"
local cca_hs_brand "..\temp\cca_hs_brand.dta"

adopath + ..\external

tempfile temp      

**********************************************************
*  MAKE CCA-HOMESCAN CROSSWALK MANY TO ONE
**********************************************************

u `cross_br', clear

** If one or more "best matches" exist for a CCA brand, keep only those **
replace best_match = 0 if best_match==.
egen has_best_match = max(best_match), by(brand_cca category_cca)
drop if has_best_match==1 & best_match==0

** When CCA brand still matches to multiple HS modules, choose the one with the most HS hhs making purchases **
mmerge module using ../external/by_module, type(n:1) unm(master) ukeep(hh_all) urename(hh_all modhh)

preserve
keep brand_cca category_cca module modhh
duplicates drop
rankunique modhh, by(brand_cca category_cca) gen(modrank)
keep if modrank==1
save `temp', replace
restore

mmerge brand_cca category_cca module using `temp', type(n:1) unmatched(none) ukeep()

** When CCA brand still matches to multiple HS brands, choose the one with the most HS hhs making purchases **
mmerge brand using ../external/by_brand, type(n:1) unm(master) ukeep(hh) 
rankunique hh, by(brand_cca category_cca) gen(brandrank)
keep if brandrank==1
drop _merge

keep brand_cca category_cca brand module
save ..\temp\crosswalk, replace

**********************************************************
*  COLLAPSE CCA TO BRAND-STATE LEVEL
**********************************************************
u `cca', clear

drop if market=="IL"
ren share_fill share_cca_orig

* merge brand
mmerge brand_cca category_cca using ..\temp\crosswalk, type(n:1) unmatched(none) ukeep(brand)

* when HS brand matches multiple CCA brands, average within brand-market-year
collapse (mean) share_cca_orig, by(brand market state year)


* collapse to brand-state level (over markets and years)
egen tagyear = tag(year state brand)
collapse (mean) share_cca_orig (sum) yrs_cca = tagyear, by(state brand)
save ..\temp\cca_brand_state, replace
 
**********************************************************
*  COLLAPSE HOMESCAN TO BRAND-STATE LEVEL
**********************************************************
u `hh_homescan', clear
mmerge hhld_id using "`demos'", type(n:1) unmatched(none) ukeep(state)

gen hhs_hs = 1
collapse (sum) purch hhs_hs, by(state brand)
replace purch = 0 if purch == .
gen module = floor(brand/10^6)
save ..\temp\hs_brand_state, replace

**********************************************************
*  COMBINE HOMESCAN AND CCA
**********************************************************
u ..\temp\hs_brand_state, clear
mmerge state brand using ..\temp\cca_brand_state, type(1:1) unmatched(both) ukeep(share_cca_orig yrs_cca)
drop if _merge==1 // homescan brands with no CCA match
drop if _merge==2 // brand-state pairs that have no purchases in homescan
drop _m

**********************************************************
* DEFINE SHARES & RANKS
**********************************************************
egen tot_hs = total(purch), by(state module)
gen share_hs = purch/tot_hs
egen tot_cca = total(share_cca_orig), by(state module)
gen share_cca = share_cca_orig/tot_cca
drop tot_hs tot_cca share_cca_orig

rankunique share_hs, by(module state) gen(rank_hs_st)
rankunique share_cca, by(module state) gen(rank_cca_st)

**********************************************************
* SAVE AT THE BRAND-STATE LEVEL FOR ALL MATCHED BRANDS (cca_hs_brand.dta)
*********************************************************
save `cca_hs_brand', replace




