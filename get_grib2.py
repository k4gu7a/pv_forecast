import urllib.request
import calendar
import os
import time

start = time.perf_counter()

def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            with open(dst_path, "wb") as local_file:
                local_file.write(web_file.read())

        # print(f"success: {dst_path}")

    except urllib.error.HTTPError as e:
        print(f"Failed: {url}")
        print(f"http error: {e.code}")

    except urllib.error.URLError as e:
        print(f"connection error: {url}")
        print(e.reason)


def generate_filename(year, month ,day):
    filerange = ("FH16-33", "FH34-39", "FH40-51")
    filenames = []
    for h_range in filerange:
        filenames.append("Z__C_RJTD_{}{}{}120000_MSM_GPV_Rjp_Lsurf_{}_grib2.bin".format(year, month, day, h_range))
    return filenames

base_url = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
base_path = os.getcwd().replace("\\","/") + "/data/msm-s_auto"

year = 2026

for i in range(1, 7):

    month = f"{i:02d}"

    for j in range(1, calendar.monthrange(year, i)[1] + 1):

        day = f"{j:02d}"

        target_file = generate_filename(year, month, day)
        target_folder = "{}/{}/{}".format(base_path, year, month)

        os.makedirs(target_folder, exist_ok=True)

        for f_name in target_file:

            target_url = "{}/{}/{}/{}/{}".format(base_url, year, month, day, f_name)
            target_path = os.path.join(target_folder, f_name)

            download_file(target_url, target_path)

end = time.perf_counter()
print('{:.2f}'.format((end-start)/60))