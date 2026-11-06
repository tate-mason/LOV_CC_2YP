* Empirical records must remain on the research server.
if "`c(os)'" != "Unix" {
    display as error "Server-only empirical workflow: do not run on this computer."
    exit 603
}
capture confirm file "/scratch/dtm63837/Kilts_Panel"
* Require the server filesystem via a directory change, then restore the working directory.
local policy_cwd "`c(pwd)'"
capture cd "/scratch/dtm63837/Kilts_Panel"
if _rc {
    display as error "Research server filesystem is required."
    exit 603
}
cd "`policy_cwd'"
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
