import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import japanize_matplotlib

s_0723 = xr.open_dataset("data/msm-s_kyo/0723.nc", cache=False)

# print(s_0723.attrs)
# print(s_0723["dswrf"].attrs)
# print(s_0723["dswrf"].encoding)

ds_dswrf = s_0723["dswrf"]
# print(ds_dswrf)

colors = ['white', 'blue', 'green', 'yellow', 'orange', 'red']
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)

fig1 = plt.figure(figsize=(12, 8))
ax1 = plt.axes(projection=ccrs.PlateCarree())

ax1.coastlines(resolution="10m", color="black", linewidth=1)
ax1.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
ax1.add_feature(cfeature.LAND, edgecolor="black")
ax1.add_feature(cfeature.LAKES, facecolor='none', edgecolor='black')
# ax1.add_feature(cfeature.RIVERS, facecolor='none', edgecolor='black')
ax1.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--")

ax1.set_extent([120, 150, 22.4, 47.6], crs=ccrs.PlateCarree())

# ax1.set_extent([134.25, 138.25, 33.28, 37.28], crs=ccrs.PlateCarree())

ims = []

# print("1")

n = 0

for t,dswrf_time in ds_dswrf.groupby("time"):

    n += 1
    if n % 3 == 1: continue

    im = dswrf_time.plot(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0, vmax=1000,
        add_labels=False,
        add_colorbar=False
    )
    ims.append([im])

    # print("2")

cbar = fig1.colorbar(
    ims[0][0],
    ax=ax1,
    orientation="vertical",
    pad=0.05
)
cbar.set_label("日射量予測値 [W/m²]")

plt.title("2026/07/23 0:00:00-23:00:00 日射量予測値 [W/m^2] ", fontsize=16)

# print("3")

ani = animation.ArtistAnimation(fig1, ims, interval=300, blit=True, repeat_delay=0)
# print("4")
ani.save("sample0.gif", writer="pillow")
# print("5")
plt.show()