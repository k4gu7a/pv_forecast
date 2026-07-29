import subprocess
import calendar
import time
import os
from pathlib import Path

start = time.perf_counter()

def generate_filename(year, month ,day):
    filerange = ("FH16-33", "FH34-39", "FH40-51")
    filenames = []
    for h_range in filerange:
        filenames.append("Z__C_RJTD_{}{}{}120000_MSM_GPV_Rjp_Lsurf_{}_grib2.bin".format(year, month, day, h_range))
    return filenames

base_path = os.getcwd().replace("\\","/") + "/data/msm-s_auto"
output_folder = os.getcwd().replace("\\","/") + "/data/msm-s_auto/nc"

os.makedirs(output_folder, exist_ok=True)

year = 2026

variables = (":(UGRD|VGRD|TMP|RH|LCDC|MCDC|HCDC|TCDC|APCP|DSWRF):")

for i in range(1, 7):

    month = f"{i:02d}"

    for j in range(1, calendar.monthrange(year, i)[1] + 1):

        day = f"{j:02d}"

        target_file = generate_filename(year, month, day)
        grib2_folder = "{}/{}/{}".format(base_path, year, month)

        os.makedirs(grib2_folder, exist_ok=True)

        k = 0

        for f_name in target_file:

            k += 1

            grib2_path = Path(os.path.join(grib2_folder, f_name))
            nc_path = Path(os.path.join(output_folder, "{}{}{}_{}.nc".format(year, month, day, k)))

            temp_grib = nc_path.with_suffix(".small.grib2")

            try:
                subprocess.run(
                    [
                        "wgrib2",
                        str(grib2_path),
                        "-match",
                        variables,
                        "-small_grib",
                        "139.0000:139.1250",
                        "35.7500:35.8500",
                        str(temp_grib),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

                subprocess.run(
                    [
                        "wgrib2",
                        str(temp_grib),
                        "-netcdf",
                        str(nc_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

            except subprocess.CalledProcessError as error:
                nc_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"convert failed: {grib2_path}"
                ) from error

            finally:
                temp_grib.unlink(missing_ok=True)

end = time.perf_counter()
print('{:.2f}'.format((end-start)/60))

# print(subprocess.run("dir", shell=True))