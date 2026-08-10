/**********************************************************
 *
 * PRECLEAN.DO
 *
 **********************************************************/
cap log close
set linesize 255

*****************************************************************
* PRELIMINARIES
*****************************************************************
version 10
clear all
set mem 500m
set matsize 800
set more off

tempfile brandlist outtemp topbrands

*****************************************************************
* LIST OF IRI CATEGORIES
*****************************************************************
local FILES : dir "../external/" files "*br_mkt_wk.csv"

*****************************************************************
* IRI DATA -- LOOP OVER CATEGORIES
*****************************************************************
local counter = 1
foreach FIL in `FILES' {

	insheet market brand week revenue volume featurevol displayvol reducevol upcs avail catvol share xldate sasdate using ../external/`FIL', clear c names
	drop if brand=="PRIVATE LABEL"
	egen tsls = sum(volume), by(brand)
	gen year = year(sasdate)
	gen month = month(sasdate)
	gen period = (year-2001)*12 + month

	gen category = "`FIL'"
	replace category = subinstr(category,"_br_mkt_wk.csv","",1)
	levelsof category, local(CAT)
	if `CAT'=="beer" {
		replace brand = "MILLER" if strpos(brand,"MILLER")>0
		replace brand = "BUDWEISER" if strpos(brand,"BUD")>0 | strpos(brand,"BDWSR")>0
	}
	if `CAT'=="blades" {
		replace brand = "GILLETTE" if strpos(brand,"GILLETTE")>0
		replace brand = "SCHICK" if strpos(brand,"SCHICK")>0
		replace brand = "BIC" if strpos(brand,"BIC")>0
	}
	if `CAT'=="cigets" {
		replace brand = "MARLBORO" if strpos(brand,"MARLBORO")>0
		replace brand = "CAMEL" if strpos(brand,"CAMEL")>0
	}
	if `CAT'=="coffee" {
		replace brand = "FOLGERS" if strpos(brand,"FOLGERS")>0
		replace brand = "MAXWELL HOUSE" if strpos(brand,"MAXWELL HOUSE")>0
	}
	if `CAT'=="deod" {
		replace brand = "MENNEN" if strpos(brand,"MENNEN")>0
		replace brand = "RIGHT GUARD" if strpos(brand,"RIGHT GUARD")>0
		replace brand = "DEGREE" if strpos(brand,"DEGREE")>0
		replace brand = "SECRET" if strpos(brand,"SECRET")>0
	}
	if `CAT'=="diapers" {
		replace brand = "PAMPERS" if strpos(brand,"PAMPERS")>0
		replace brand = "HUGGIES" if strpos(brand,"HUGGIES")>0
	}
	if `CAT'=="factiss" {
		replace brand = "KLEENEX" if strpos(brand,"KLEENEX")>0
		replace brand = "SCOTTIES" if strpos(brand,"SCOTTIES")>0
		replace brand = "PUFFS" if strpos(brand,"PUFFS")>0
	}
	if `CAT'=="margbutr" {
		replace brand = "Shedd's Country Crock" if strpos(brand,"SHEDD'S")>0
		replace brand = "I CAN'T BELIEVE IT'S NOT BUTTER" if strpos(brand,"I CAN'T BELIEVE")>0
	}
	if `CAT'=="mayo" {
		replace brand = "KRAFT" if strpos(brand,"KRAFT")>0 | strpos(brand,"MIRACLE WHIP")>0
		replace brand = "HELLMANNS" if strpos(brand,"HELLMANNS")>0 | strpos(brand,"BEST FOODS")>0
	}
	if `CAT'=="mustketc" {
		replace brand = "HEINZ" if strpos(brand,"HEINZ")>0
		replace brand = "HUNTS" if strpos(brand,"HUNTS")>0
	}
	if `CAT'=="peanbutr" {
		replace brand = "JIF" if strpos(brand,"JIF")>0
		replace brand = "SKIPPY" if strpos(brand,"SKIPPY")>0
	}
	if `CAT'=="razors" {
		replace brand = "GILLETTE" if strpos(brand,"GILLETTE")>0
		replace brand = "SCHICK" if strpos(brand,"SCHICK")>0
	}
	if `CAT'=="shamp" {
		replace brand = "ALBERTO" if strpos(brand,"ALBERTO")>0
		replace brand = "SUAVE" if strpos(brand,"SUAVE")>0
		replace brand = "CLAIROL" if strpos(brand,"CLAIROL")>0
	}
	if `CAT'=="spagsauc" {
		replace brand = "RAGU" if strpos(brand,"RAGU")>0
		replace brand = "PREGO" if strpos(brand,"PREGO")>0
	}
	if `CAT'=="toitisu" {
		replace brand = "QUILTED" if strpos(brand,"QUILTED")>0
		replace brand = "CHARMIN" if strpos(brand,"CHARMIN")>0
		replace brand = "KLEENEX" if strpos(brand,"KLEENEX")>0
		replace brand = "ANGEL SOFT" if strpos(brand,"ANGEL SOFT")>0
	}
	if `CAT'=="toothbr" {
		replace brand = "ORAL B" if strpos(brand,"ORAL B")>0
		replace brand = "COLGATE" if strpos(brand,"COLGATE")>0
		replace brand = "CREST" if strpos(brand,"CREST")>0
	}
	if `CAT'=="toothpa" {
		replace brand = "COLGATE" if strpos(brand,"COLGATE")>0
		replace brand = "CREST" if strpos(brand,"CREST")>0
	}
	if `CAT'=="yogurt" {
		replace brand = "YOPLAIT" if strpos(brand,"YOPLAIT")>0
		replace brand = "DANNON" if strpos(brand,"DANNON")>0
	}

	*** COLLAPSE DATA TO WEEKLY FREQUENCY (i.e. aggregate up variants of brands) ***
	*** NOTE: we volume-weight the average upcs and availability across the sub-brands ***
	replace upcs = upcs*volume
	replace avail = avail*volume
	collapse (sum) volume revenue featurevol displayvol reducevol upcs avail, by(brand market year month period week)
	replace upcs = upcs/volume
	replace avail = avail/volume
	gen price = revenue/volume
	gen feature = featurevol/volume
	gen display = displayvol/volume
	gen reduce = reducevol/volume

	*** RETAIN BRANDS WITH COMPLETE GEOGRAPHIC COVERAGE ***
	preserve
		collapse (mean) volume, by(brand market)
		gen count=1
		collapse (sum) count, by(brand)
		keep if count==47
		sort brand
		save `brandlist', replace
	restore

	sort brand
	merge brand using `brandlist'
	keep if _m==3
	drop _m


	*** RETAIN TOP TWO BRANDS IN THE CATEGORY (NATIONAL LEVEL) ***
	preserve
		collapse (mean) volume, by(brand)
		gsort -volume
		gen rank = _n==1
		replace rank = 2 if _n==2
		gen firstbrand = brand if rank==1
		gen secondbrand = brand if rank==2
		keep if _n<=2
		keep brand rank first second
		sort brand
		save `topbrands', replace
	restore

	mmerge brand using `topbrands', type(n:1) unmatched(none) missing(none)
	keep market week brand volume price feature display reduce upcs avail rank revenue year month period

	*** RESHAPE DATA TO CREATE RELATIVE SHARES AND PRICES ***
	reshape wide volume price feature display reduce upcs avail brand revenue, i(market week) j(rank)
	gen category = "`FIL'"
	replace category = regexr(category,"_br_mkt_wk.csv","")

	*** calculate share of top 2 sales and other measures for regressions ***
	gen share1 = volume1/(volume1+volume2)
	gen share2 = volume2/(volume1+volume2)
	gen y = log(share1)-log(share2)
	gen dlprice = log(price1)-log(price2)
	gen dfeat = feature1-feature2
	gen ddisp = display1-display2
	gen dred = reduce1-reduce2
	gen dupc = upcs1-upcs2
	gen davail = avail1-avail2

	keep if share1~=. & share2~=.

	*** POOL CATEGORY DATA ***	
	if `counter'==1 {
		save ../temp/pooldata, replace
	}
	if `counter'>1 {
		save `outtemp', replace
		use ../temp/pooldata, clear
		append using `outtemp'
		save ../temp/pooldata, replace
	}

	local counter = `counter'+1

}

cap log close
