import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe


def show_fig_size_on_resize(fig):
    def on_resize(event):
        # Get the updated figure size
        new_size = event.canvas.figure.get_size_inches()
        print(f"Figure resized: {new_size[0]:.2f} x {new_size[1]:.2f} inches")

    fig.canvas.mpl_connect("resize_event", on_resize)


def create_mesh(ax, lx=4.0, ly=3.0, nx=4, ny=3):
    # hard-coded parameters
    num_elems_x = nx
    num_elems_y = ny

    lw = 1.0
    markersize = 30

    # Derived parameters
    hx = lx / num_elems_x
    hy = ly / num_elems_y

    num_points_x = num_elems_x + 1
    num_points_y = num_elems_y + 1
    # Create the x and y coordinates of the grid
    x = np.linspace(-lx / 2.0, lx / 2.0, num_points_x)
    y = np.linspace(-ly / 2.0, ly / 2.0, num_points_y)

    # Create the meshgrid
    X, Y = np.meshgrid(x, y)

    ax.plot(X, Y, "k-", lw=lw, zorder=5, clip_on=False)  # Horizontal lines
    ax.plot(X.T, Y.T, "k-", lw=lw, zorder=5, clip_on=False)  # Vertical lines

    # level set function
    N_lsf = 200
    X_lsf = np.linspace(-lx / 2.0, lx / 2.0, N_lsf)
    Y_lsf = np.linspace(-ly / 2.0, ly / 2.0, N_lsf)
    X_lsf, Y_lsf = np.meshgrid(X_lsf, Y_lsf)

    def lsf_func(x, y):
        return (x**2 + 2.0 * y**2 - 0.5 * x * y - (0.45 * lx) ** 2.0) * (
            (x + 0.11 * lx) ** 2 + (y + 0.06 * ly) ** 2 - (0.2 * lx) ** 2.0
        )

    def lsf_func_vert(v):
        x = v[0] / nx * lx - 0.5 * lx
        y = v[1] / ny * ly - 0.5 * ly
        return lsf_func(x, y)

    # Loop over cells
    active_cells = []
    active_verts = []
    cut_cells = []
    for ic in range(nx):
        for jc in range(ny):
            v1, v2, v3, v4 = (ic, jc), (ic, jc + 1), (ic + 1, jc), (ic + 1, jc + 1)
            elem_verts_lsf_vals = np.array([lsf_func_vert(v) for v in [v1, v2, v3, v4]])
            if (elem_verts_lsf_vals < 0.0).any():
                active_verts.extend([v1, v2, v3, v4])
                active_cells.append([ic, jc])
            if elem_verts_lsf_vals.min() * elem_verts_lsf_vals.max() < 0.0:
                cut_cells.append([ic, jc])

    inner_label = False
    cut_label = False
    for c in active_cells:
        if c not in cut_cells:
            rectangle_inner = patches.Rectangle(
                (X[c[1], c[0]], Y[c[1], c[0]]),
                hx,
                hy,
                edgecolor="none",
                facecolor="blue",
                alpha=0.3,
                clip_on=False,
            )
            ax.add_patch(rectangle_inner)
            if not inner_label:
                rectangle_inner.set_label("inner CGD elements")
                inner_label = True

        else:
            rectangle_cut = patches.Rectangle(
                (X[c[1], c[0]], Y[c[1], c[0]]),
                hx,
                hy,
                edgecolor="none",
                facecolor="red",
                alpha=0.3,
                clip_on=False,
            )
            ax.add_patch(rectangle_cut)
            if not cut_label:
                rectangle_cut.set_label("cut CGD elements")
                cut_label = True

    Xdof = []
    Ydof = []
    for v in active_verts:
        x = v[0] / nx * lx - 0.5 * lx
        y = v[1] / ny * ly - 0.5 * ly
        Xdof.append(x)
        Ydof.append(y)

    ax.scatter(
        Xdof,
        Ydof,
        facecolor="gray",
        edgecolor="black",
        s=markersize,
        lw=lw,
        zorder=10,
        clip_on=False,
        label="active CGD dof nodes",
    )

    lsf = ax.contour(
        X_lsf,
        Y_lsf,
        lsf_func(X_lsf, Y_lsf),
        [0.0],
        colors="black",
        linewidths=lw,
    )
    pe_list = [pe.withTickedStroke(angle=135, length=1, spacing=5)]
    lsf.set_path_effects(pe_list)

    custom_legend = plt.Line2D([0], [0], color="black", path_effects=pe_list)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(custom_legend)
    labels.append("level set boundary")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.axis("off")

    return handles, labels


# Plot the mesh
fig, ax = plt.subplots(
    ncols=1,
    nrows=1,
    figsize=(6.64, 4.80),
)
show_fig_size_on_resize(fig)

handles, labels = create_mesh(ax=ax, lx=15.0, ly=10.0, nx=15, ny=10)

# Show the plot
l = fig.legend(
    handles,
    labels,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    bbox_transform=fig.transFigure,
)
l.get_frame().set_edgecolor("none")
l.get_frame().set_facecolor("none")

plt.subplots_adjust(
    left=0.03, bottom=0.03, right=0.97, top=0.88, wspace=0.0, hspace=0.0
)
fig.savefig("cut_mesh_example.jpg", dpi=1200)
plt.show()
