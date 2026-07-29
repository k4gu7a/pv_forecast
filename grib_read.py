import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import japanize_matplotlib

lsurfdata = xr.open_dataset("data/meps_nc/Lsurf_sample.nc", cache=False)

# print(lsurfdata)

# variables = lsurfdata.data_vars
# print("\n変数一覧:")
# print(variables)

# for var_name, var in variables.items():
#     print(f"\n変数名: {var_name}")
#     print(var)

sliced_data = lsurfdata.sel(time='2019-06-05T6:00:00')

dswrf_surface = sliced_data["DSWRF_surface"]
tmp_10m = sliced_data["TMP_1D5maboveground"]

colors = ['white', 'blue', 'green', 'yellow', 'orange', 'red']
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)

fig1 = plt.figure(figsize=(12, 8))
ax1 = plt.axes(projection=ccrs.PlateCarree())

dswrf_surface.plot(
    ax=ax1,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmin=0, vmax=1000,
    cbar_kwargs={"label": "日射量 [W/m^2]"}
)

ax1.coastlines(resolution="10m", color="black", linewidth=1)
ax1.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
ax1.add_feature(cfeature.LAND, edgecolor="black")
ax1.add_feature(cfeature.LAKES, facecolor='none', edgecolor='black')
ax1.add_feature(cfeature.RIVERS, facecolor='none', edgecolor='black')
ax1.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--")

ax1.set_extent([134.25, 138.25, 33.28, 37.28], crs=ccrs.PlateCarree())

plt.title("日射量予測値 [W/m^2] 2019-06-05 6:00:00 UTC", fontsize=16)
plt.show()

fig2 = plt.figure(figsize=(12, 8))
ax2 = plt.axes(projection=ccrs.PlateCarree())

tmp_10m.plot(
    ax=ax2,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmin=278, vmax=308,
    cbar_kwargs={"label": "気温 [K]"}
)

ax2.coastlines(resolution="10m", color="black", linewidth=1)
ax2.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
ax2.add_feature(cfeature.LAND, edgecolor="black")
ax2.add_feature(cfeature.LAKES, facecolor='none', edgecolor='black')
ax2.add_feature(cfeature.RIVERS, facecolor='none', edgecolor='black')
ax2.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--")

ax2.set_extent([134.25, 138.25, 33.28, 37.28], crs=ccrs.PlateCarree())

plt.title("気温予測値 [K] 2019-06-05 6:00:00 UTC", fontsize=16)
plt.show()