/**********************************************************
 *
 * CROSS_MARKET.DO
 *
 **********************************************************/
cap log close
set linesize 255

*****************************************************************
* PRELIMINARIES
*****************************************************************
version 11
clear all
set mem 500m
set matsize 800
set more off

tempfile results

adopath + ../external

*****************************************************************
* DESCRIPTIVE ANALYSIS OF PRICE VARIATION
*****************************************************************
use ../temp/pooldata, clear
keep if week==1373

gen lnrelprice = log(price1/price2)
gen lnrelshare = log(volume1/volume2)
gen share = volume1/(volume1+volume2)

* look at prices & quantities for coffee
sort category market
list market price1 price2 volume1 volume2 share if category=="coffee"

scatter lnrelprice lnrelshare if category=="coffee", mlabel(market)
graph export "../output/coffee_xmkt.eps", replace

* regressions by category
postfile res str20(category) double(coeff se R2) using `results', replace
levelsof category, local(catlist)
foreach CAT in `catlist' {
	reg lnrelshare lnrelprice if category=="`CAT'"
	post res ("`CAT'") (_b[lnrelprice]) (_se[lnrelprice]) (e(r2))
}
postclose res

* pooled regression
encode market, gen(nmarket)
areg lnrelshare lnrelprice i.nmarket, absorb(category)

* output posted results
use `results', clear
outsheet using ../output/regressions_xmkt.csv, replace comma noquote

*****************************************************************
* DECOMPOSING CROSS-MARKET VARIATION IN SHARES
*****************************************************************
use ../temp/pooldata, clear

* Compute spatial variation by category
preserve
collapse (mean) y dupc davail, by(category market)
egen spatial = sd(y), by(category)
egen medspatial = median(spatial)
gen str4 spatialstatus = "high" if spatial>=medspatial
replace spatialstatus = "low" if spatial<medspatial
collapse (mean) dupc davail, by(spatialstatus)
list
restore

* regressions by category
postfile res str20(category variable) double(corr R2 R2_cns) using `results', replace

encode market, gen(nmarket)
levelsof category, local(catlist)
foreach CAT in `catlist' {
	preserve
	keep if category=="`CAT'"
	display("Category: `CAT'")

	foreach V in dlprice dfeat ddisp dupc davail{
		display("Variable: `V'")
		areg y `V' i.nmarket, absorb(week) cluster(market)
		scalar b`V' = _b[`V']
	}
	areg y dlprice dfeat ddisp i.nmarket, absorb(week)
    scalar bpfd_dlprice = _b[dlprice]
    scalar bpfd_dfeat = _b[dfeat]
    scalar bpfd_ddisp = _b[ddisp]
    	
	collapse (mean) y dlprice dfeat ddisp dupc davail, by(market)

	sum y
	scalar yvar = r(Var)

	foreach V in dlprice dfeat ddisp dupc davail {

		display("Variable: `V'")

		reg y `V'
		scalar R2 = e(r2)
		scalar corr = sqrt(e(r2))*sign(_b[`V'])

        gen pred_`V' = `V'*b`V'
		sum pred_`V'
		scalar R2_cns = r(Var)/yvar
		
		post res ("`CAT'") ("`V'") (corr) (R2) (R2_cns)
	}

    display("Variable: PFD")

	gen pred_pfd = dlprice*bpfd_dlprice + dfeat*bpfd_dfeat + ddisp*bpfd_ddisp
    sum pred_pfd
	scalar R2_cns = r(Var)/yvar
	post res ("`CAT'") ("pfd") (.) (.) (R2_cns)
    
	display("Variable: All")

	reg y dlprice dfeat ddisp dupc davail
	scalar corr = sqrt(e(r2))
	scalar R2 = e(r2)
	post res ("`CAT'") ("all") (corr) (R2) (.)
    
	restore
}
postclose res

* output posted results
use `results', clear
outsheet using ../output/forecast_by_category.csv, replace comma noquote

collapse (mean) corr R2 R2_cns (sd) sdcorr=corr sdR2=R2 sdR2_cns=R2_cns, by(variable)
mkmat corr R2 R2_cns sdcorr sdR2 sdR2_cns, rownames(variable) matrix(OUT) 
matrix_to_txt, matrix(OUT) saving(../output/forecast_sum.txt) replace user usec

save ../temp/cross_market.dta, replace
cap log close
