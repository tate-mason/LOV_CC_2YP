/**********************************************************
 *
 * ONLINE_APPENDIX.DO
 *
 **********************************************************/

cap log close
log using ..\output\online_appendix.log, text replace
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

cap erase ..\output\online_appendix_tables.txt


*****************************************************************
* Estimates for subsample of state-modules for which CCA data
* confirms past-present gap in shares was small (R2, point 8)
*****************************************************************

tempfile temp
insheet using ../external/rel_shares.csv
keep if ps_hs<. & ps_cca<.
gen absdif = abs(ps_hs-ps_cca)
save `temp', replace

u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(state_born state_curr years years_cat age_move age_move_cat)

* keep only state-modules that exist in CCA
mmerge state_born module using `temp', type(n:1) unm(none) umatch(state module) ukeep(absdif) urename(absdif absdif_born)
mmerge state_curr module using `temp', type(n:1) unm(none) umatch(state module) ukeep(absdif) urename(absdif absdif_curr)
gen decades = years/10
gen decades2 = decades^2
drop if betahat == . 

reg betahatnd decades decades2 [aw=sqdifnd], cluster(module)
test (_cons=.624) (decades=.098) (decades2=-.009)

reg betahatnd decades decades2 [aw=sqdifnd] if absdif_born<.2 & absdif_curr<.2, cluster(module)
test (_cons=.624) (decades=.098) (decades2=-.009)

reg betahatnd decades decades2 [aw=sqdifnd] if absdif_born<.1 & absdif_curr<.1, cluster(module)
test (_cons=.624) (decades=.098) (decades2=-.009) 


*****************************************************************
* Alternative years & ages figures with 
* non-parametric controls (R1, point 4)
*****************************************************************

u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years_cat age_move_cat age_move years age)
quietly tab years_cat, gen(zyears)
quietly tab age_move_cat, gen(zagemove)

reg betahatnd zyears1-zyears13 zagemove2-zagemove13 [aw=sqdifnd], cluster(module) nocons

local xlab "0-4 5-9 10-14 15-19 20-24 25-29 30-34 35-39 40-44 45-49 50-54 55-59 60+" 
local options "ylabel(0(.1)1) yline(0 1, lpattern(dash)) yscale(range(-.1 1.1)) ytitle(Relative Share ({&beta} {subscript:ij})) xtitle(Years Since Move) label(`xlab')"
plotcoeffs zyears1-zyears13, `options'
graph export ..\output\reply\years_none_alt.eps, as(eps) replace

local xlab "5-9 10-14 15-19 20-24 25-29 30-34 35-39 40-44 45-49 50-54 55-59 60+" 
local options "ylabel(-0.5(.1)0.5) yscale(range(-.6 .6)) ytitle(Relative Share ({&beta} {subscript:ij})) xtitle(Age at Move) label(`xlab')"
plotcoeffs zagemove2-zagemove13 , `options'
graph export ..\output\reply\agemove_none_alt.eps, as(eps) replace

*****************************************************************
* SAMPLE SPLITS
*  - Demographics (R1, point 7)
*  - Difference between origin & destination state (R1, point 11)
*  - Impact of secondary household member (R2, point 1)
*****************************************************************

* Save HH characteristics
u ..\external\survey.dta, clear

* define_secondary_is_migrant {
    egen numinhh = total(1), by(hhld_id)
    keep if numinhh<=2
    egen nummig = total(state_born~=state_curr), by(hhld_id)
    egen primmig = total(keeper & state_born~=state_curr), by(hhld_id)
    gen secmig = nummig-primmig>0
    egen born_curr_code = group(state_born state_curr)
    qui unique born_curr_code, by(hhld_id) gen(temp)
    egen num_born_curr_code = max(temp), by(hhld_id)
    
    gen secondary_is_migrant = 0 if numinhh==2 & primmig==1 & secmig==0
    replace secondary_is_migrant = 1 if nummig==2 & num_born_curr_code==1
    local secondary_is_migrant_lab0 "Secondary HH member is non-migrant"
    local secondary_is_migrant_lab1 "Secondary HH member is migrant"
* }

gen education_college_plus = educ>=6 if keeper
local education_college_plus_lab0 "Primary HH member less than college degree"
local education_college_plus_lab1 "Primary HH member college degree or more"

gen income_over_median = inc>55000 if keeper
local income_over_median_lab0 "HH income $55,000 or below"
local income_over_median_lab1 "HH income over $55,000"

local male_lab0 "Primary HH member is female"
local male_lab1 "Primary HH member is male"

local survey_chars "secondary_is_migrant education_college_plus income_over_median male"

keep if keeper
keep hhld_id `survey_chars'
tempfile temp_hh
save `temp_hh', replace

* Save module characteristics
use ..\temp\stddev, clear
sum sdstateshare, det
gen sdstateshare_over_median = sdstateshare>r(p50)
local sdstateshare_over_median_lab0 "Cross-state variation below median"
local sdstateshare_over_median_lab1 "Cross-state variation above median"

tempfile temp_mod
save `temp_mod', replace

* Save HH-module characteristics
use ..\temp\purch_mig, clear
sum sqdifnd, det
gen sqdif_over_median = sqdifnd>r(p50)
local sqdif_over_median_lab0 "Difference below median"
local sqdif_over_median_lab1 "Difference above median"

tempfile temp_hhmod
save `temp_hhmod', replace

* Analysis
u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years_cat age_move_cat age_move years age)
mmerge hhld_id using `temp_hh', type(n:1) unm(master) ukeep(`survey_chars')
mmerge hhld_id module using `temp_hhmod', type(n:1) unm(master) ukeep(sqdif_over_median)
mmerge module using `temp_mod', type(n:1) unm(master) ukeep(sdstateshare_over_median)

gen years_bin = 1 if years<=15
replace years_bin = 2 if years>15 & years<=30
replace years_bin = 3 if years>30 & years<=45
replace years_bin = 4 if years>45

local options "ylabel(0(.1)1) yline(0 1, lpattern(dash)) yscale(range(-.1 1.1)) ytitle(Relative Share ({&beta} {subscript:ij})) xtitle(Years Since Move)"
    
foreach VAR in `survey_chars' sqdif_over_median sdstateshare_over_median {

    preserve    
    drop if `VAR'==.
    
    * Table
    foreach BIN of numlist 1/4 {
        qui reg betahatnd ibn.`VAR' [w=sqdifnd] if years_bin==`BIN', nocons
        test 0.`VAR' = 1.`VAR'
        matrix TABLE = (nullmat(TABLE) \ ((_b[0.`VAR'] \ _se[0.`VAR']) , (_b[1.`VAR'] \ _se[1.`VAR']) , (r(p) \ . )))
    }
    matrix_to_txt, saving(..\output\online_appendix_tables.txt) mat(TABLE) format(%20.6f) title(<tab:`VAR'>) append
    cap matrix drop TABLE
	
    * Figure
    collapse (mean) betahatnd [w=sqdifnd], by(years_cat `VAR')
    twoway scatter betahatnd years_cat if `VAR'==0, msize(small) || /// 
       scatter betahatnd years_cat if `VAR'==1, msize(small) `options' ///
       legend(label(1 "``VAR'_lab0'") ///
       label(2 "``VAR'_lab1'") cols(1)) ///
	   xlabel(0 "0-4" 5 "5-9" 10 "10-14" 15 "15-19" 20 "20-24" 25 "25-29" 30 "30-34" 35 "35-39" 40 "40-44" 45 "45-49" 50 "50-54" 55 "55-59" 60 "60+", angle(90))
	   graph export ..\output\figures\\`VAR'.eps, as(eps) replace
    
    restore
}


*****************************************************************
* Placebo version of Table 4 (R2, point 7)
*****************************************************************
cap program drop addtotable_bp
program addtotable_bp
	local r2 = e(r2)
	local N = e(N)
	local prs = e(N_clust)
	nlcom (par1: _b[years_treated]) (par2: _b[treat_in_b]) (par3: _b[years_untreated]) (par4: _b[not_treat_in_b]), post
	matrix TABLE = (nullmat(TABLE) , (_b[par1] \ _se[par1] \ _b[par2] \ _se[par2] \ _b[par3] \ _se[par3] \ _b[par4] \ _se[par4] \ `prs' \ `N'))
end

* generate random version of yearlaunch
u ..\temp\purch_mig_bp, clear
keep pairid launch_year1 launch_year2
duplicates drop
gen yearlaunch = min(launch_year1, launch_year2)
gen urv = runiform()
qui sum yearlaunch
gen yearlaunch_rand = floor(r(min) + urv*(r(max) - r(min)))
tempfile temp_rand
save `temp_rand', replace

u ..\temp\purch_mig_bp, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years)
mmerge pairid using `temp_rand', type(n:1) unm(master) ukeep(yearlaunch_rand)

* generate bunch of variables plus weights
gen yearumbrella = min(umbrella_year1, umbrella_year2)  
gen years_since_launch = 2007-yearlaunch_rand
gen years_b = years_since_launch - years
gen treat_in_b = years_b>0
gen not_treat_in_b = 1-treat_in_b
gen years_untreated = not_treat_in_b*years 
gen years_treated = treat_in_b*years 
gen w1 = totpurch*sqdif/(pairshare*(1-pairshare))
gen w1nd = totpurch*sqdifnd/(pairshare*(1-pairshare))

* data screen drop if betahat == .
drop if betahat == . 

* all pairs (yearlaunch_rand>=1955)
reg betahatnd treat_in_b years_treated not_treat_in_b years_untreated [aw=sqdifnd] , cluster(pairid) nocons
addtotable_bp

* post-1975
reg betahatnd treat_in_b years_treated not_treat_in_b years_untreated [aw=sqdifnd] if yearlaunch_rand>=1975, cluster(pairid) nocons
addtotable_bp

* post-1985
reg betahatnd treat_in_b years_treated not_treat_in_b years_untreated [aw=sqdifnd] if yearlaunch_rand>=1985, cluster(pairid) nocons
addtotable_bp

matrix_to_txt, saving(..\output\online_appendix_tables.txt) mat(TABLE) format(%20.6f) title(<tab:Years_bp_placebo>) append

*****************************************************************
* FGLS VERSION OF DESCRIPTIVE MODEL (<Tab:Years>)
*****************************************************************
u ..\temp\purch_mig, clear
mmerge hhld_id using ..\temp\hh_mig, type(n:1) unm(master) ukeep(years years_cat age_move age_move_cat)
cap matrix drop TABLE
gen decades = years/10
gen decades2 = decades^2
egen avgshare = mean(share), by(module)
gen sampling_variance = avgshare*(1-avgshare)/totpurch
drop if betahat == . 
gen age_move_10 = age_move/10

* decades only
reg betahatnd decades decades2 [aw=sqdifnd], cluster(module)
predict e, resid
gen e2 = e^2
reg e2 sampling_variance
predict e2hat
reg betahatnd decades decades2 [aw=sqdifnd/e2hat], cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (_b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ . \ . \ _b[_cons] \ _se[_cons] \ `mod' \ `N'))
drop e e2 e2hat

* decades & age move
reg betahatnd decades decades2 age_move_10 [aw=sqdifnd], cluster(module)
predict e, resid
gen e2 = e^2
reg e2 sampling_variance
predict e2hat
reg betahatnd decades decades2 age_move_10 [aw=sqdifnd/e2hat], cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (_b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ _b[age_move_10] \ _se[age_move_10] \ _b[_cons] \ _se[_cons] \ `mod' \ `N'))
drop e e2 e2hat

* dummies for age_move
gen age_move_temp = max(age_move,0)
reg betahatnd decades decades2 i.age_move_temp  [aw=sqdifnd], cluster(module)
predict e, resid
gen e2 = e^2
reg e2 sampling_variance
predict e2hat
reg betahatnd decades decades2 i.age_move_temp [aw=sqdifnd/e2hat], cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (_b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ . \ . \ . \ . \ `mod' \ `N'))
drop e e2 e2hat

* dummies for years
reg betahatnd age_move_10 i.years  [aw=sqdifnd], cluster(module)
predict e, resid
gen e2 = e^2
reg e2 sampling_variance
predict e2hat
reg betahatnd age_move_10 i.years [aw=sqdifnd/e2hat], cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (. \ . \ . \ . \ _b[age_move_10] \ _se[age_move_10] \ . \ . \ `mod' \ `N'))
drop e e2 e2hat

* only those moving after age 25
reg betahatnd decades decades2 age_move_10 [aw=sqdifnd] if age_move>=25, cluster(module)
predict e, resid
gen e2 = e^2
reg e2 sampling_variance
predict e2hat
reg betahatnd decades decades2 age_move_10 [aw=sqdifnd/e2hat] if age_move>=25, cluster(module)
local N = e(N)
local mod = e(N_clust)
matrix TABLE = (nullmat(TABLE) , (_b[decades] \ _se[decades] \ _b[decades2] \ _se[decades2] \ _b[age_move_10] \ _se[age_move_10] \ _b[_cons] \ _se[_cons] \ `mod' \ `N'))

matrix_to_txt, saving(..\output\online_appendix_tables.txt) mat(TABLE) format(%20.6f) title(<tab:YearsFGLS>) append
cap matrix drop TABLE
cap log close

