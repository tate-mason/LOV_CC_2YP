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
version 11
clear
set mem 2G
set matsize 5000
set more off
adopath + ..\external\

cap erase ..\output\texttables.txt

*****************************************************************
* MAKE TABLES FOR TEXT.PDF
*****************************************************************
use "..\temp\text.dta", clear
summarize absdiv 
local absdiv_mean = r(mean)
summarize large
local large_mean = r(mean)

matrix TABLE = (nullmat(TABLE),(`absdiv_mean' \ `large_mean'))
matrix_to_txt, saving(..\output\texttables.txt) mat(TABLE) format(%20.6f) title(<tab:change_purchase_shares>) replace
matrix drop TABLE

cap log close
