from pathlib import Path
import xarray as xr
import os

ncs_folder = os.getcwd().replace("\\","/") + "/data/msm-s_auto/nc"
output_path = os.getcwd().replace("\\","/") + "/data/msm-s_auto/all.nc"

files = sorted(Path(ncs_folder).glob("*.nc"))

ds = xr.open_mfdataset(
    files,
    combine="by_coords"
)

ds.to_netcdf(output_path)