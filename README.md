# rstarGlobal-18countries

“The Fall 2025 Brookings Papers on Economic Activity *["Discussion of 'What next for r*' by Lukasz Rachel"]() by Marco Del Negro extends the Global r* model in *[Global Trends in Interest Rates](https://doi.org/10.1016/j.jinteco.2019.01.010)* by Marco Del Negro, Domenico Giannone, Marc Giannoni, and Andrea Tambalotti to 18 countries.


# Update
The `update` folder contains CSV files of updated estimates of the trends in r* using data on short- and long-term yields, and inflation, computed using the rstarGlobal model. The US/Global r* (figure above) and 18 country-specific r* (figure below) figures show updated estimates of r*.

**US/Global r\***  
![US/Global r*](https://github.com/FRBNY-DSGE/rstarGlobal-18countries/blob/master/update/qRshort_bar_us_global_m1.png)  
*Source: Author's calculations. The dashed black line shows the posterior median of global r* and the shaded areas show the 68 and 95 percent posterior coverage intervals. The dotted black line shows the posterior median of U.S. r\**  

**18 Country-specific r\***  
![18 Country-specific r*](https://github.com/FRBNY-DSGE/rstarGlobal-18countries/blob/master/update/qRshort_bar_m2.png)  
*Source: Author's calculations. The chart plots the posterior median estimates of r* for the 18 countries in the sample.*


## Required software

These scripts were produced using MATLAB R2020a.

## Replicating figures and tables

[*“The Fall 2025 Brookings Papers on Economic Activity: Discussion of ‘What next for r*?’ by Łukasz Rachel,” discussed by Marco Del Negro*]() uses Model 1, which estimates the baseline model with prior to variance of innovation to the trend component equal to 1, and Model 2, which estimates the model with a convenience yield. The mentioned figures are referenced to in this codebase as 'Figure 1' and 'Figure 3b'.

- load updated data into data file: `indata/DataInflShortLongUpdated.xlsx`
- run Model 1
	- run the Matlab script: `scripts/MainModel.m`
	- model outputs outputted to `results/18/OutputModel1.mat`
- run Model 2
	- run the Matlab script: `scripts/MainModel2.m`
	- model outputs outputted to `results/18/OutputModel2.mat`
- create figures
	- to recreate figures from this discussion: in `scripts/MainModel1_MakeFigures.m`, run the 'Plot preliminaries' block, then 'Figure 1' block, then 'Figure 3b' block
- create table
	- to recreate the table from this discussion: in `scripts/makeTables.m`, run the 'Table A1a: Change in r-bar^w_t and Its Components (baseline), additional details' block to get the baseline change, then run the 'Table A1b: Change in r-bar^w_t and Its Components (Convenience Yield), additional details' block to get the change under the convenience yield model, then .tex tables are outputted to `tables/`

## Data sources

The [`indata`](indata/) folder contains the underlying data. Data on short-term rates, long-term rates, and consumer prices come from the Jordà-Schularick-Taylor Macrohistory Database. Data for Moody's Baa corporate bond yield are available from FRED. Demographic data come from the United States Census Bureau and from the UN World Population Statistics database. Data on corporate spreads come from Gilchrist and Mojon (2018). Please see *[Global Trends in Interest Rates](https://doi.org/10.1016/j.jinteco.2019.01.010)* by Marco Del Negro, Domenico Giannone, Marc Giannoni, and Andrea Tambalotti for additional details on the data sources.

## Disclaimer
Copyright Federal Reserve Bank of New York. You may reproduce, use, modify, make derivative works of, and distribute and this code in whole or in part so long as you keep this notice in the documentation associated with any distributed works. Neither the name of the Federal Reserve Bank of New York (FRBNY) nor the names of any of the authors may be used to endorse or promote works derived from this code without prior written permission. Portions of the code attributed to third parties are subject to applicable third party licenses and rights. By your use of this code you accept this license and any applicable third party license.

THIS CODE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EXCEPT TO THE EXTENT THAT THESE DISCLAIMERS ARE HELD TO BE LEGALLY INVALID. FRBNY IS NOT, UNDER ANY CIRCUMSTANCES, LIABLE TO YOU FOR DAMAGES OF ANY KIND ARISING OUT OF OR IN CONNECTION WITH USE OF OR INABILITY TO USE THE CODE, INCLUDING, BUT NOT LIMITED TO DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, PUNITIVE, SPECIAL OR EXEMPLARY DAMAGES, WHETHER BASED ON BREACH OF CONTRACT, BREACH OF WARRANTY, TORT OR OTHER LEGAL OR EQUITABLE THEORY, EVEN IF FRBNY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES OR LOSS AND REGARDLESS OF WHETHER SUCH DAMAGES OR LOSS IS FORESEEABLE.
