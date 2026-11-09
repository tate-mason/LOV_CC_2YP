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
 * TABLES.DO
 *
 **********************************************************/

cap log close
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
* TABLE <Tab:forecasting>
*****************************************************************
use ../output/forecast, clear
keep if category=="Mean" | category=="St. Dev."

mkmat rhoprice corrrelprice rhofeat corrrelfeat rhodisp corrreldisp rhoupc corrrelupc rhoavail corrrelavail, mat(TABLE)

matrix_to_txt, saving(..\output\tables.txt) mat(TABLE) format(%20.3f) title(<tab:forecasting>) replace
