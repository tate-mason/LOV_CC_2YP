/**********************************************************
 *
 * TEXT.DO: Produces supporting facts for Brands paper.
 *
 **********************************************************/

cap log close
cap log using "../output/text.log", text replace
set linesize 255

*****************************************************************
* PRELIMINARIES
*****************************************************************
version 10
clear all
set mem 1g
set matsize 5000
set more off
adopath + ..\external\

cap erase ..\output\texttables.txt
loadglob using ..\temp\figures.txt

*****************************************************************
* SAMPLE DESCRIPTION - TIMEFRAME OF PURCHASE DATA
*****************************************************************
use ..\external\hh_module_month.dta,clear
format ym %10.0g
gen date = dofm(ym)
format date %d
gen month = month(date)
gen year = year(date)
sort ym
local yearstart = year[1]
local monthstart = month[1]
quietly count
local yearend= year[`r(N)']
local monthend = month[`r(N)']

matrix TABLE = (nullmat(TABLE),(`yearstart', `monthstart' \ `yearend', `monthend'))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:samp_year>) append
matrix drop TABLE

*****************************************************************
* SAMPLE DESCRIPTION - SIZE OF FINAL DATA (NUM OF HOUSEHOLDS)
*****************************************************************

use ..\temp\hh_mig.dta,clear
sort hhld_id
by hhld_id: gen uni=_n==1
count if uni==1
scalar temp1=r(N)
use ..\temp\hh_nonmig.dta,clear
sort hhld_id
by hhld_id: gen uni=_n==1
count if uni==1
matrix TABLE = (nullmat(TABLE),(temp1+r(N)))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:num_hhld_final>) append
drop uni
mat drop TABLE


*****************************************************************
* SAMPLE DESCRIPTION - PURCHASE HISTORY
*****************************************************************

use ..\external\by_hh_brand.dta,clear
sort hhld_id
by hhld_id: gen uni = _n==1
count if uni==1
scalar num_hhld_id = r(N)
drop uni

sort brand
by brand: gen uni = _n==1
count if uni==1
scalar num_brand = r(N)
drop uni

sort module
by module: gen uni = _n==1
count if uni==1
scalar num_module = r(N)
matrix TABLE = (nullmat(TABLE),(num_hhld_id\num_brand\num_module))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:desc_purch_hist>) append
drop uni
mat drop TABLE

*****************************************************************
* SAMPLE DESCRIPTION - SURVEY
*****************************************************************

use ..\\external\survey.dta,clear
sort hhld_id
by hhld_id: gen uni= _n==1
count if uni==1
scalar num_hhld_survey = r(N)
quietly tab state_curr
scalar num_state = r(r)
matrix TABLE = (nullmat(TABLE),(num_state\num_hhld_survey\_N\75221\(num_hhld_survey/75221)))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:desc_survey>) append
drop uni
mat drop TABLE

*****************************************************************
* SAMPLE DESCRIPTION - PERCENTAGE MOVE (TO DIFFERENT REGION)
*****************************************************************

use ..\\temp\hh_nonmig.dta,clear
scalar num_nonmig = _N
use ..\\temp\hh_mig.dta,clear
scalar num_mig = _N
gen temp=1 if reg_born!=reg_curr
su temp
matrix TABLE = (nullmat(TABLE),(r(N)/(num_nonmig+num_mig)))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:perc_move>) append
mat drop TABLE

*****************************************************************
* STATISTICAL TESTS FOR RECENT MOVERS
*****************************************************************

u ..\temp\purch_mig_month, clear
gen year = year(dofm(ym))
gen month = month(dofm(ym))
quietly tab ym, gen(zym)

keep if (years==1 | years==0) & gap==0

* note: survey fielded 9/13/08 - 10/1/08
drop if year==2008 & month>=10
gen early = ((year==2007 & month<=9) | year==2006)
gen late = ((year==2007 & month>9) | year==2008)
gen w1nd = totpurch*sqdifnd/(modshare*(1-modshare))

unique hhld_id if years==0
unique hhld_id if years==1
quietly tab hhld_id if years==0
local zeroyear `r(r)'
quietly tab hhld_id if years==1
local oneyear `r(r)'

matrix TABLE = (nullmat(TABLE),(`zeroyear' \ `oneyear'))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:samp_which_year>) append
matrix drop TABLE

reg betahatnd early late if years==0 [aw=sqdifnd], cluster(module) nocons
test early=0
test early=$alpha
test early=late



* Output table for regression results - years == 0

scalar cil_early = _b[early]-invttail(e(df_r),.025)*_se[early]
scalar ciu_early = _b[early]+invttail(e(df_r),.025)*_se[early]
scalar ciu_late = _b[late]+invttail(e(df_r),.025)*_se[late]
scalar cil_late = _b[late]-invttail(e(df_r),.025)*_se[late]

matrix TABLE = (nullmat(TABLE),(_b[early] \ _b[late]))
matrix TABLE = (nullmat(TABLE),(_se[early] \ _se[late]))
matrix TABLE = (nullmat(TABLE),(_b[early]/_se[early] \ _b[late]/_se[late]))
matrix TABLE = (nullmat(TABLE),(ttail(e(df_r),(_b[early]/_se[early]))*2 \ (ttail(e(df_r),_b[late]/_se[late]))*2))
matrix TABLE = (nullmat(TABLE),( cil_early \ cil_late))
matrix TABLE = (nullmat(TABLE),( ciu_early \ ciu_late))

matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:reg_early_late>) append
matrix drop TABLE

reg betahatnd early late if years==1 [aw=sqdifnd], cluster(module) nocons
test late=0
test late=$alpha
test early=late

reg betahatnd age_move if years==0 & early [aw=w1nd], cluster(module)
matrix TABLE = (nullmat(TABLE),(_b[age_move]))
matrix TABLE = (nullmat(TABLE),(_se[age_move]))
matrix TABLE = (nullmat(TABLE),(_b[age_move]/_se[age_move]))
matrix TABLE = (nullmat(TABLE),(ttail(e(df_r),_b[age_move]/_se[age_move])*2))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:reg_age_move>) append
matrix drop TABLE

reg betahatnd age_move if years==1 & late [aw=w1nd], cluster(module)

*****************************************************************
* Section 5 in the paper - secondary assumption test 
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
test years_treated=0
scalar w1_1955=r(p)
test years_untreated=0
scalar w3_1955=r(p)
test not_treat_in_b=1
scalar w2_1955=r(p)
test years_untreated=1-not_treat_in_b=0
scalar w2w3_1955=r(p)

* post-1975
reg betahatnd treat_in_b years_treated not_treat_in_b years_untreated [aw=sqdifnd] if yearlaunch>=1975, cluster(pairid) nocons
test years_treated=0
scalar w1_1975=r(p)
test years_untreated=0
scalar w3_1975=r(p)
test not_treat_in_b=1
scalar w2_1975=r(p)
test years_untreated=1-not_treat_in_b=0
scalar w2w3_1975=r(p)

* post-1985
reg betahatnd treat_in_b years_treated not_treat_in_b years_untreated [aw=sqdifnd] if yearlaunch>=1985, cluster(pairid) nocons
test years_treated=0
scalar w1_1985=r(p)
test years_untreated=0
scalar w3_1985=r(p)
test not_treat_in_b=1
scalar w2_1985=r(p)
test years_untreated=1-not_treat_in_b=0
scalar w2w3_1985=r(p)

matrix TABLE = (nullmat(TABLE),(w1_1955\w3_1955\w2_1955\w2w3_1955))
matrix TABLE = (nullmat(TABLE),(w1_1975\w3_1975\w2_1975\w2w3_1975))
matrix TABLE = (nullmat(TABLE),(w1_1985\w3_1985\w2_1985\w2w3_1985))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:move_before_brands_introduced>) append
matrix drop TABLE
*********************************************************************************************
* Section 7 in the paper - correlation between dummy for advertising and that for visibility
*********************************************************************************************

use ..\temp\modchar.dta,clear
reg social ad
scalar corr_social_ad = sign(_b[ad])*sqrt(e(r2))
matrix TABLE = (nullmat(TABLE),(corr_social_ad))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:corr_social_ad>) append

cap log close
