# PV forecast
## Data preprocessing
### MSM (*grib2.bin → *.nc)
`get_grib2.py → grib2_to_netcdf.py → concat_msm.py`

### PV status (*.csv)
`request_pv_status.py`

## For testing 
* accuracy_comparison.py
* grib_read.py
* nc_to_pred.py

## References
* 京大生存圏研究所 
  * http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/
* PVOutput
  * https://pvoutput.org/about.html
* About MSM
  * https://www.jmbsc.or.jp/jp/online/file/f-online10200.html