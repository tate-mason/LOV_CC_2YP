/**********************************************************
 *
 * FORECAST_IRI.DO
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

tempfile catdata relpriceout relfeatout reldispout relupcout relavailout relpriceout_panel relfeatout_panel reldispout_panel relupcout_panel relavailout_panel forecast1

*****************************************************************
* IRI DATA -- LOOP OVER CATEGORIES
*****************************************************************
use ../temp/pooldata, clear

local counter = 1
levelsof category, local(CATS)
local flag=1
foreach CAT in `CATS' {
	preserve
	keep if category=="`CAT'"
	save `catdata', replace
		
	*** FIRST STAGE (1): REGRESSION ANALYSIS USING ONLY MONTHLY VARIATION (COLLAPSE GEOGRAPHY) ***
	collapse (sum) volume* revenue* (mean) feature* display* upc* avail*, by(period brand1 brand2)
	** MAKE VARIABLES
	gen relprice = (revenue1/volume1)/(revenue2/volume2)
	gen relfeat = feature1/feature2
	gen reldisp = display1/display2
	gen relupc = upcs1/upcs2
	gen relavail = avail1/avail2
	gen share1 = volume1/(volume1+volume2)

	statsby _b _se e(r2) e(N), saving(`relpriceout', replace): reg share1 relprice
	statsby _b _se e(r2) e(N), saving(`relfeatout', replace): reg share1 relfeat
	statsby _b _se e(r2) e(N), saving(`reldispout', replace): reg share1 reldisp
	statsby _b _se e(r2) e(N), saving(`relupcout', replace): reg share1 relupc
	statsby _b _se e(r2) e(N), saving(`relavailout', replace): reg share1 relavail


	*** FIRST STAGE (2): REGRESSION ANALYSIS USING PANEL VARIATION ***
	use `catdata', clear
	** MAKE VARIABLES
	gen relprice = (revenue1/volume1)/(revenue2/volume2)
	gen relfeat = feature1/feature2
	gen reldisp = display1/display2
	gen relupc = upcs1/upcs2
	gen relavail = avail1/avail2
	xi i.market
	
	statsby _b _se e(r2) e(N), saving(`relpriceout_panel', replace): areg share1 relprice _I*, absorb(week)
	statsby _b _se e(r2) e(N), saving(`relfeatout_panel', replace):  areg share1 relfeat _I*, absorb(week)
	statsby _b _se e(r2) e(N), saving(`reldispout_panel', replace):  areg share1 reldisp _I*, absorb(week)
	statsby _b _se e(r2) e(N), saving(`relupcout_panel', replace):	 areg share1 relupc _I*, absorb(week)
	statsby _b _se e(r2) e(N), saving(`relavailout_panel', replace): areg share1 relavail _I*, absorb(week)

	
	*** SECOND STAGE: FORECASTING EXERCISE USING ONLY GEOGRAPHIC VARIATION ***
	use `catdata', clear

	collapse (sum) volume* revenue* (mean) feature* display* upc* avail*, by(market category brand1 brand2)
	** MAKE VARIABLES
	gen relprice = (revenue1/volume1)/(revenue2/volume2)
	gen relfeat = feature1/feature2
	gen reldisp = display1/display2
	gen relupc = upcs1/upcs2
	gen relavail = avail1/avail2
	gen share1 = volume1/(volume1+volume2)

	*** RAW GEOGRAPHIC CORRELATIONS ***
	corr share1 relprice
	gen rhoprice = r(rho)
	corr share1 relfeat
	gen rhofeat = r(rho)
	corr share1 reldisp
	gen rhodisp = r(rho)
	corr share1 relupc
	gen rhoupc = r(rho)
	corr share1 relavail
	gen rhoavail = r(rho)
	
	** MERGE REGRESSION COEFFICIENTS BACK WITH MASTER DATA
	** FIRST USE TIME-SERIES ONLY
	local VARS "relprice relfeat reldisp relupc relavail"
	quietly {
		foreach VAR in `VARS' {
			merge using ``VAR'out'
			drop _m
			egen cons`VAR' = max(_b_cons)
			egen beta`VAR' = max(_b_`VAR')
			drop _b*

			*** FORECASTING EXERCISE ***
			gen shhat_`VAR' = cons`VAR' + beta`VAR'*`VAR'
			reg share1 shhat_`VAR'
			gen rmse`VAR' = e(rmse)
			corr share1 shhat_`VAR'
			gen corr`VAR' = r(rho)
		}
	}	

	** SECOND USE PANEL VARIATION
	local VARS "relprice relfeat reldisp relupc relavail"
	quietly {
		foreach VAR in `VARS' {
			merge using ``VAR'out_panel'
			drop _m
			egen cons`VAR'_panel = max(_b_cons)
			egen beta`VAR'_panel = max(_b_`VAR')
			drop _b*

			*** FORECASTING EXERCISE ***
			gen shhat_`VAR'_panel = cons`VAR'_panel + beta`VAR'_panel*`VAR'
			reg share1 shhat_`VAR'_panel
			gen rmse`VAR'_panel = e(rmse)
			corr share1 shhat_`VAR'_panel
			gen corr`VAR'_panel = r(rho)
		}
	}	

	
	keep category brand1 brand2 corr* rmse* rho*
	keep if _n==1
	*** APPEND ***
	if `flag'==1 {
		cd ../output
		save `forecast1', replace
		cd ../code
	}
	else {
		cd ../output
		append using `forecast1'
		save `forecast1', replace
		cd ../code
	}
	local flag=0
	restore
}

cd ../output
use `forecast1', clear

sort category
save ../output/forecast, replace


** STANDARD DEVIATIONS OF RESULTS
preserve
collapse (sd) corr* rmse* rho*
gen category = "St. Dev."
save ../output/forecast, replace
restore


** MEANS OF RESULTS
preserve
collapse (mean) corr* rmse* rho*
gen category = "Mean"
append using ../output/forecast
save ../output/forecast, replace
restore

collapse (mean) corr* rmse* rho*, by(category brand1 brand2)
append using ../output/forecast
save ../output/forecast, replace

local VARS "relprice relfeat reldisp relupc relavail"
foreach VAR in `VARS' {
	replace corr`VAR' = round( corr`VAR'*1000 )/1000
	replace rmse`VAR' = round( rmse`VAR'*1000 )/1000
	replace corr`VAR'_panel = round( corr`VAR'_panel*1000 )/1000
	replace rmse`VAR'_panel = round( rmse`VAR'_panel*1000 )/1000
}


outsheet category brand1 brand2 rhoprice corrrelprice rmserelprice rhofeat corrrelfeat rmserelfeat rhodisp corrreldisp rmsereldisp rhoupc corrrelupc rmserelupc rhoavail corrrelavail rmserelavail using ../output/forecast.csv, c replace
outsheet category brand1 brand2 rhoprice corrrelprice_panel rmserelprice_panel rhofeat corrrelfeat_panel rmserelfeat_panel rhodisp corrreldisp_panel rmsereldisp_panel rhoupc corrrelupc_panel rmserelupc_panel rhoavail corrrelavail_panel rmserelavail_panel using ../output/forecast_panel.csv, c replace
cd ../code

cap log close
