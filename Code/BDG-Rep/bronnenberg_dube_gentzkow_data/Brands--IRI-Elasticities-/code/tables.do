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
