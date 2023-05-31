import matplotlib.pyplot as plt


def plot_choro_mp(gdf, col: str):
    vmin = 0
    vmax = gdf[col].max()
    ax = gdf.plot(column=col, figsize=(11, 8), vmin=vmin, vmax=vmax, cmap="viridis")

    # add colorbar
    fig = ax.get_figure()
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    fig.colorbar(sm, cax=cax)
