/**********************************************************
 *
 * TEXT.DO: Produces supporting facts for Brands paper.
 *
 **********************************************************/

cap log close
set linesize 255

*****************************************************************
* PRELIMINARIES
*****************************************************************
version 10
clear all
set mem 1000m
set matsize 5000
set more off
adopath + ..\external\

cap erase ..\output\texttables.txt


*****************************************************************
* POOLED ANALYSIS
*****************************************************************
use ../temp/pooldata, clear
encode category, gen(ncategory)
egen market_category = group(market category)

unique market
unique category
unique week

quietly tab market
local marketnum `r(r)'
quietly tab category
local categorynum `r(r)'
quietly tab week
local weeknum `r(r)'

matrix TABLE = (nullmat(TABLE),(`marketnum' \ `categorynum' \ `weeknum'))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:data_count>) replace
matrix drop TABLE

areg y dlprice c.dfeat#ncategory c.ddisp#ncategory, absorb(market_category) cluster(market)
local coeff = _b[dlprice]
local se = _se[dlprice]

matrix TABLE = (nullmat(TABLE),(`coeff' \ `se'))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:elast_of_subs>) append
matrix drop TABLE

areg share1 dlprice c.dfeat#ncategory c.ddisp#ncategory, absorb(market_category) cluster(market)
local coeff = _b[dlprice]
local se = _se[dlprice]

insheet using "../external/tables.txt", clear
keep if v1[_n-1] == "<Tab:Struct>"
keep v1
destring v1, replace
local alpha = v1
local implied_gamma = `coeff'/`alpha'

matrix TABLE = (nullmat(TABLE),(`coeff' \ `se' \ `implied_gamma'))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:price_effect>) append
matrix drop TABLE

************************************************************************************
* CROSS MARKET CORRELATION BETWEEN REL.SHARE AND PRICE, FEATURES, AND DISPLAY
************************************************************************************
use ../temp/cross_market.dta, clear
mkmat corr R2 R2_cns sdcorr sdR2 sdR2_cns, matrix(TABLE)
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:cross_market_corr>) append
matrix drop TABLE

cap log close
