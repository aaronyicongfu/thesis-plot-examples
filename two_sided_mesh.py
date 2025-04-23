import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe


markersize = 15.0
lw = 1.0


def show_fig_size_on_resize(fig):
    def on_resize(event):
        # Get the updated figure size
        new_size = event.canvas.figure.get_size_inches()
        print(f"Figure resized: {new_size[0]:.2f} x {new_size[1]:.2f} inches")

    fig.canvas.mpl_connect("resize_event", on_resize)


def plot_cut_mesh(
    ax3d,
    lx=1.5,
    ly=1.0,
    nx=15,
    ny=10,
    zoffset=0.0,
    lsf_func=lambda x, y: 1.0 - x * x - y * y,
):

    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    X, Y = np.meshgrid(x, y)

    def lsf_func_vert(v):
        x = v[0] / nx * lx
        y = v[1] / ny * ly
        return lsf_func(x, y)

    x_shared = []
    y_shared = []

    active_verts = []

    # Loop over cells
    for ic in range(nx):
        for jc in range(ny):
            x_v = [
                X[jc][ic],
                X[jc + 1][ic],
                X[jc + 1][ic + 1],
                X[jc][ic + 1],
                X[jc][ic],
            ]
            y_v = [
                Y[jc][ic],
                Y[jc + 1][ic],
                Y[jc + 1][ic + 1],
                Y[jc][ic + 1],
                Y[jc][ic],
            ]

            v1, v2, v3, v4 = (ic, jc), (ic, jc + 1), (ic + 1, jc), (ic + 1, jc + 1)
            elem_verts_lsf_vals = np.array([lsf_func_vert(v) for v in [v1, v2, v3, v4]])

            # cell (ic, jc) is a cut_cell
            if elem_verts_lsf_vals.min() * elem_verts_lsf_vals.max() < 0.0:
                x_shared.extend(x_v)
                y_shared.extend(y_v)

            # cell (ic, jc) is an active cell
            if (elem_verts_lsf_vals < 0.0).any():
                active_verts.extend([v1, v2, v3, v4])

                ax3d.plot(
                    x_v,
                    y_v,
                    zs=zoffset,
                    zdir="z",
                    color="black",
                    lw=lw,
                )

    ax3d.set_axis_off()
    ax3d.set_aspect("equal")

    xlim = ax3d.get_xlim()
    ylim = ax3d.get_ylim()
    zlim = ax3d.get_zlim()

    lsf = ax3d.contour(
        X, Y, lsf_func(X, Y), levels=[0], colors="blue", offset=zoffset, linewidths=lw
    )
    pe_list = [pe.withTickedStroke(angle=155, length=1, spacing=5)]
    lsf.set_path_effects(pe_list)

    Xdof = []
    Ydof = []
    active_verts = list(set(active_verts))
    for v in active_verts:
        x = v[0] / nx * lx
        y = v[1] / ny * ly
        Xdof.append(x)
        Ydof.append(y)

    sp = ax3d.scatter(
        Xdof,
        Ydof,
        np.ones_like(Xdof) * zoffset,
        facecolor="gray",
        edgecolor="black",
        s=markersize,
        lw=lw,
        zorder=10,
        clip_on=False,
        alpha=1.0,
    )

    if zoffset == 0.0:
        sp.set_label("DOF nodes")

    ax3d.set_xlim(xlim)
    ax3d.set_ylim(ylim)
    ax3d.set_zlim(zlim)
    ax3d.view_init(azim=-112.0, elev=17.0, roll=0)

    return np.array(x_shared), np.array(y_shared)


# parameters
lx = 1.2
ly = 0.8
nx = 12
ny = 8
r = 0.68
zoffset = 0.15

smoke = False
if smoke:
    nx = 3
    ny = 2


lsf_func = lambda x, y: r * r - x * x - y * y
lsf_func_inverted = lambda x, y: x * x + y * y - r * r

fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax3d = fig.add_subplot(111, projection="3d")


plot_cut_mesh(ax3d, lx=lx, ly=ly, nx=nx, ny=ny, zoffset=zoffset, lsf_func=lsf_func)
x1, y1 = plot_cut_mesh(
    ax3d, lx=lx, ly=ly, nx=nx, ny=ny, zoffset=0.0, lsf_func=lsf_func_inverted
)

# Dedup
xd = list(set([(xv, yv) for xv, yv in zip(x1, y1)]))
x1, y1 = np.array(xd).T

# Connect overlapping nodes
for xval, yval in zip(x1, y1):
    ax3d.plot(
        xs=[xval, xval],
        ys=[yval, yval],
        zs=[0.0, zoffset],
        ls="--",
        lw=0.5 * lw,
        color="blue",
    )

# Set legends
pe_list = [pe.withTickedStroke(angle=155, length=1, spacing=5)]
handles, labels = ax3d.get_legend_handles_labels()
handles.insert(0, plt.Line2D([0], [0], color="blue", lw=lw, path_effects=pe_list))
labels.insert(0, "level set boundaries")
handles.insert(2, plt.Line2D([0], [0], color="blue", lw=lw * 0.5, ls="--"))
labels.insert(2, "overlapping DOF nodes")
l = ax3d.legend(
    handles,
    labels,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.72),
    bbox_transform=fig.transFigure,
)
l.get_frame().set_edgecolor("none")
l.get_frame().set_facecolor("none")

fig.savefig("two_sided_mesh.jpg", dpi=1200, bbox_inches="tight")
plt.show()
