/**********************************************************
 *
 * TABLES.DO
 *
 **********************************************************/

cap log close
log using ..\output\tables.log, text replace
set linesize 255

**********************************************************
* PRELIMINARIES
**********************************************************
version 11
clear all
set mem 500m
set matsize 5000
set more off
set seed 04271975
set sortseed 04271975
adopath + ..\external

cap erase ..\output\tables.txt

*****************************************************************
* TABLE <Tab:Migration>
*****************************************************************
u hhld_id reg_born reg_curr using ..\temp\hh_nonmig, clear
append using ..\temp\hh_mig, keep(hhld_id reg_born reg_curr)

* for reference
tab reg_born reg_curr

* hack because Stata won't let you get at the results from a tabulate command!
cap matrix drop TABLE
foreach B of numlist 1/4 {
	cap matrix drop ROW
	foreach C of numlist 1/4 {
		quietly count if reg_born==`B' & reg_curr==`C'
		matrix ROW = (nullmat(ROW),r(N))
	}
	matrix TABLE = (nullmat(TABLE) \ ROW)
}

matrix_to_txt, saving(..\output\tables.txt) mat(TABLE) format(%20.3f) title(<tab:Migration>) append

*****************************************************************
* TABLE <Tab:SumStat>
*****************************************************************

* # of modules
insheet using ..\external\sample_mod.csv, names comma clear
count
scalar modules = r(N)

* # of non-migrants
u ..\temp\hh_nonmig, clear
count
scalar nonmig = r(N)

* # of migrants
u ..\temp\hh_mig, clear
count
scalar mig = r(N)

* avg purchases & share
u hhld_id module purch1 purch2 share using ..\temp\purch_nonmig, clear
append using ..\temp\purch_mig, keep(hhld_id module purch1 purch2 share)
sum purch1
scalar purch1 = r(mean)
sum purch2
scalar purch2 = r(mean)
sum share
scalar avgshare = r(mean)

* sdstateshare
tempfile temp_statemod
use ..\temp\purch_nonmig, clear
mmerge hhld_id using ..\temp\hh_nonmig, type(n:1) unm(master) ukeep(state_curr)
collapse (mean) stateshare = share, by(module state_curr)
save `temp_statemod', replace
collapse (sd) sdstateshare = stateshare, by(module)
sum sdstateshare
scalar sdstateshare = r(mean)

* avgabsdif (muc - mub)
u  ..\temp\purch_mig, clear
gen absdif = abs(muc - mub)
sum absdif
scalar avgabsdif = r(mean)

cap matrix drop TABLE
matrix TABLE = (nullmat(TABLE) \ modules \ nonmig \ mig \ purch1 \ purch2 \ avgshare \ sdstateshare \ avgabsdif)
matrix_to_txt, saving(..\output\tables.txt) mat(TABLE) format(%20.6f) title(<tab:SumStat>) append

*****************************************************************
* TABLE <Tab:Years>
*****************************************************************
u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years years_cat age_move age_move_cat)
cap matrix drop TABLE
gen decades = years/10
gen decades2 = decades^2
gen w1 = totpurch*sqdif/(modshare*(1-modshare))
gen w1nd = totpurch*sqdifnd/(modshare*(1-modshare))

* data screen drop if betahat == .
drop if betahat == . 

* rescale age_move coefficients
gen age_move_10 = age_move/10

* decades only
reg betahatnd decades decades2 [aw=sqdifnd], cluster(module)
	* save "on-impact" effect of moving (alpha) in temp file for recent-movers figures
	file open figures using "..\temp\figures.txt", write replace
	file write figures "alpha" _skip(1) (_b[_cons]) _n
	file close figures
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (_b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ . \ . \ _b[_cons] \ _se[_cons] \ `mod' \ `N'))

* decades & age move
reg betahatnd decades decades2 age_move_10 [aw=sqdifnd], cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (_b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ _b[age_move_10] \ _se[age_move_10] \ _b[_cons] \ _se[_cons] \ `mod' \ `N'))

* dummies for age_move
gen age_move_temp = max(age_move,0)
reg betahatnd decades decades2 i.age_move_temp [aw=sqdifnd], cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (_b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ . \ . \ . \ . \ `mod' \ `N'))

* dummies for years
reg betahatnd age_move_10 i.years [aw=sqdifnd], cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (. \ . \ . \ . \ _b[age_move_10] \ _se[age_move_10] \ . \ . \ `mod' \ `N'))

* only those moving after age 25
reg betahatnd decades decades2 age_move_10 [aw=sqdifnd] if age_move>=25, cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (_b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ _b[age_move_10] \ _se[age_move_10] \ _b[_cons] \ _se[_cons] \ `mod' \ `N'))

matrix_to_txt, saving(..\output\tables.txt) mat(TABLE) format(%20.6f) title(<tab:Years>) append
cap matrix drop TABLE

*****************************************************************
* Table <Tab:Years_bp>
*****************************************************************
cap program drop addtotable_bp
program addtotable_bp
	local r2 = e(r2)
	local N = e(N)
	local prs = e(N_clust)
	nlcom (par1: _b[years_treated]) (par2: _b[treat_in_b]) (par3: _b[years_untreated]) (par4: _b[not_treat_in_b]), post
	matrix TABLE = (nullmat(TABLE) , (_b[par1] \ _se[par1] \ _b[par2] \ _se[par2] \ _b[par3] \ _se[par3] \ _b[par4] \ _se[par4] \ `prs' \ `N'))
end

u ..\temp\purch_mig_bp, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years)

*generate bunch of variables plus weights
gen yearumbrella = min(umbrella_year1, umbrella_year2) 
gen yearlaunch = min(launch_year1, launch_year2) 
gen ysl = 2007-yearlaunch
gen years_b = ysl - years
gen treat_in_b = years_b>0
gen not_treat_in_b = 1-treat_in_b
gen years_untreated = not_treat_in_b*years 
gen years_treated = treat_in_b*years 
gen w1 = totpurch*sqdif/(pairshare*(1-pairshare))
gen w1nd = totpurch*sqdifnd/(pairshare*(1-pairshare))

* data screen drop if betahat == .
drop if betahat == . 

* all pairs (yearlaunch>=1955)
reg betahatnd treat_in_b years_treated not_treat_in_b years_untreated [aw=sqdifnd] , cluster(pairid) nocons
addtotable_bp

* post-1975
reg betahatnd treat_in_b years_treated not_treat_in_b years_untreated [aw=sqdifnd] if yearlaunch>=1975, cluster(pairid) nocons
addtotable_bp

* post-1985
reg betahatnd treat_in_b years_treated not_treat_in_b years_untreated [aw=sqdifnd] if yearlaunch>=1985, cluster(pairid) nocons
addtotable_bp

matrix_to_txt, saving(..\output\tables.txt) mat(TABLE) format(%20.6f) title(<tab:Years_bp>) append


*****************************************************************
* TABLE <Tab:Hetero>
*****************************************************************
cap program drop addtotable
program addtotable
	local r2 = e(r2)
	local N = e(N)
	local mod = e(N_clust)
	nlcom (cons: _b[_cons]) (coeff: _b[decades]) (years: 10*(1-_b[_cons])/_b[decades]), post
	matrix TABLE = (nullmat(TABLE) , (_b[cons] \ _se[cons] \ _b[coeff] \ _se[coeff] \ _b[years] \ _se[years] \ `r2' \ `mod' \ `N'))
end

u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years age_move_cat)
mmerge module using ..\temp\modchar, type(n:1) unm(master) ukeep(ad social)
cap matrix drop TABLE
gen decades = years/10

gen w1 = totpurch*sqdif/(modshare*(1-modshare))

foreach V in ad social {

	reg betahat decades if `V'==0 [aw=sqdif], cluster(module)
	addtotable
	
	reg betahat decades if `V'==1 [aw=sqdif], cluster(module)
	addtotable
	
	gen cons_low`V' = `V'==0
	gen cons_high`V' = `V'==1
	gen decades_low`V' = decades*(`V'==0)
	gen decades_high`V' = decades*(`V'==1)
	reg betahat cons_low`V' cons_high`V' decades_low`V' decades_high`V' [aw=sqdif], cluster(module) nocons
	nlcom (consdif: _b[cons_high`V']-_b[cons_low`V']) (coeffdif: _b[decades_high`V']-_b[decades_low`V']) ///
	      (yearsdif: 10*(1-_b[cons_high`V'])/_b[decades_high`V'] - 10*(1-_b[cons_low`V'])/_b[decades_low`V']), post
	matrix TABLE = (nullmat(TABLE) , (_b[consdif] \ _se[consdif] \ _b[coeffdif] \ _se[coeffdif] \ _b[yearsdif] \ _se[yearsdif] \ . \ .\ .) )

}

matrix_to_txt, saving(..\output\tables.txt) mat(TABLE) format(%20.6f) title(<tab:Hetero>) append

*****************************************************************
* TABLE (Extra regression table for slides)
*****************************************************************
cap program drop addtotable
program addtotable
	local r2 = e(r2)
	local N = e(N)
	local mod = e(N_clust)
	nlcom (cons: _b[_cons]) (coeff: _b[decades]) (years: 10*(1-_b[_cons])/_b[decades]), post
	matrix TABLE = (nullmat(TABLE) , (_b[cons] \ _se[cons] \ _b[coeff] \ _se[coeff] \ . \ . \ _b[years] \ _se[years] \ `r2' \ `mod' \ `N'))
end

u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years age_move_cat)
cap matrix drop TABLE
gen decades = years/10
gen decades2 = decades^2
gen dec_move_cat = age_move_cat/10
gen w1 = totpurch*sqdif/(modshare*(1-modshare))
gen w1nd = totpurch*sqdif/(modshare*(1-modshare))

* no controls
reg betahatnd decades decades2 dec_move_cat [aw=sqdifnd], cluster(module) 
matrix TABLE = (nullmat(TABLE) , (_b[_cons] \ _se[_cons] \ _b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ _b[dec_move_cat] \ _se[dec_move_cat]))

* controls
reg betahat decades decades2 dec_move_cat [aw=sqdif], cluster(module) 
matrix TABLE = (nullmat(TABLE) , (_b[_cons] \ _se[_cons] \ _b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ _b[dec_move_cat] \ _se[dec_move_cat]))

matrix_to_txt, saving(..\output\tables.txt) mat(TABLE) format(%20.6f) title(<extra_for_slides>) append

*****************************************************************
* APPENDIX TABLE <Tab:mod260>
*****************************************************************
* calculate aggregate purchase share & save tempfile
tempfile share
u module purch1 purch2 using ..\external\by_module.dta, clear
gen share_agg = purch1 / (purch1+purch2)
keep module share_agg
save `share', replace

* create table: restrict to 260 modules used in main analysis
insheet using ..\external\sample_mod.csv, clear

* merge original module names and top-2 brand names
mmerge module using ..\external\by_module, type(1:1) unm(none) ukeep(brand_name1 brand_name2)

* merge cleaned module and brand names
mmerge module using ..\input\module_name_display, type(1:1) unm(none) ukeep(module_name_display)
mmerge brand_name1 using ..\input\brand_name_display, type(n:1) unm(none) umatch(brand_name) ukeep(brand_name_display) urename(brand_name_display brand_name1_display)
mmerge brand_name2 using ..\input\brand_name_display, type(n:1) unm(none) umatch(brand_name) ukeep(brand_name_display) urename(brand_name_display brand_name2_display)

* merge aggregate purchase share
mmerge module using `share', type(1:1) unm(none)

* merge cross-state std dev of avg purchase share
mmerge module using ../temp/stddev.dta, type(1:1) unm(none)

* merge indicators for advertising intense, socially visible
mmerge module using ..\temp\modchar, type(1:1) unm(master) ukeep(ad social)
drop brand_name1 brand_name2 _merge
sort module_name_display

* write to tables.txt
file open tables using "..\output\tables.txt", write append

** obs 1-45
file write tables "<tab:mod260-1>" _n
foreach i of numlist 1/45 {
	file write tables (module_name_display[`i']) _tab (brand_name1_display[`i']) _tab (brand_name2_display[`i']) _tab (share_agg[`i']) _tab (sdstateshare[`i']) _tab (ad[`i']) _tab (social[`i']) _n
}

** obs 46-92
file write tables "<tab:mod260-2>" _n
foreach i of numlist 46/92 {
	file write tables (module_name_display[`i']) _tab (brand_name1_display[`i']) _tab (brand_name2_display[`i']) _tab (share_agg[`i']) _tab (sdstateshare[`i']) _tab (ad[`i']) _tab (social[`i']) _n
}

** obs 93-139
file write tables "<tab:mod260-3>" _n
foreach i of numlist 93/139 {
	file write tables (module_name_display[`i']) _tab (brand_name1_display[`i']) _tab (brand_name2_display[`i']) _tab (share_agg[`i']) _tab (sdstateshare[`i']) _tab (ad[`i']) _tab (social[`i']) _n
}

** obs 140-186
file write tables "<tab:mod260-4>" _n
foreach i of numlist 140/186 {
	file write tables (module_name_display[`i']) _tab (brand_name1_display[`i']) _tab (brand_name2_display[`i']) _tab (share_agg[`i']) _tab (sdstateshare[`i']) _tab (ad[`i']) _tab (social[`i']) _n
}

** obs 187-233
file write tables "<tab:mod260-5>" _n
foreach i of numlist 187/233 {
	file write tables (module_name_display[`i']) _tab (brand_name1_display[`i']) _tab (brand_name2_display[`i']) _tab (share_agg[`i']) _tab (sdstateshare[`i']) _tab (ad[`i']) _tab (social[`i']) _n
}

** obs 234-238
file write tables "<tab:mod260-6>" _n
foreach i of numlist 234/238 {
	file write tables (module_name_display[`i']) _tab (brand_name1_display[`i']) _tab (brand_name2_display[`i']) _tab (share_agg[`i']) _tab (sdstateshare[`i']) _tab (ad[`i']) _tab (social[`i']) _n
}

file close tables

cap log close

