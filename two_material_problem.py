import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe


import scienceplots

plt.style.use(["science"])


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

    ax.plot(
        X, Y, "-", lw=0.5, color="grey", zorder=15, clip_on=False
    )  # Horizontal lines
    ax.plot(
        X.T, Y.T, "-", lw=0.5, color="grey", zorder=15, clip_on=False
    )  # Vertical lines

    # level set function
    N_lsf = 200
    X_lsf = np.linspace(-lx / 2.0, lx / 2.0, N_lsf)
    Y_lsf = np.linspace(-ly / 2.0, ly / 2.0, N_lsf)
    X_lsf, Y_lsf = np.meshgrid(X_lsf, Y_lsf)

    def lsf_func(x, y):
        return (x**2 + 2.0 * y**2 - 0.5 * x * y - (0.45 * lx) ** 2.0) * (
            (x + 0.11 * lx) ** 2 + (y + 0.06 * ly) ** 2 - (0.2 * lx) ** 2.0
        )

    def lsf_inner(x, y):
        return (x + 0.11 * lx) ** 2 + (y + 0.06 * ly) ** 2 - (0.2 * lx) ** 2.0

    def lsf_outer(x, y):
        return x**2 + 2.0 * y**2 - 0.5 * x * y - (0.45 * lx) ** 2.0

    def lsf_func_vert(v):
        x = v[0] / nx * lx - 0.5 * lx
        y = v[1] / ny * ly - 0.5 * ly
        return lsf_func(x, y)

    ax.contour(
        X_lsf,
        Y_lsf,
        lsf_func(X_lsf, Y_lsf),
        levels=[0.0],
        colors="black",
        linewidths=1.0,
        zorder=20,
    )

    Z = lsf_inner(X_lsf, Y_lsf)
    mask = (X_lsf >= -2.0) | (Y_lsf >= -0.5)
    Z_masked = np.ma.masked_where(mask, Z)

    lsf_bcs = ax.contour(
        X_lsf,
        Y_lsf,
        Z_masked,
        levels=[0.0],
        colors="black",
        linewidths=1.0,
        zorder=20,
    )

    pe_list = [pe.withTickedStroke(angle=135, length=1, spacing=5)]
    lsf_bcs.set_path_effects(pe_list)

    Z = lsf_outer(X_lsf, Y_lsf)
    mask = (X_lsf < 3.5) | (Y_lsf < 1.0)
    Z_masked = np.ma.masked_where(mask, Z)

    lsf_loads = ax.contour(
        X_lsf,
        Y_lsf,
        Z_masked,
        levels=[0.0],
        colors="black",
        linewidths=1.0,
        zorder=20,
    )

    pe_list = [
        pe.withTickedStroke(angle=60, length=0.5, spacing=10),
        pe.withTickedStroke(angle=120, length=0.5, spacing=10),
        pe.withTickedStroke(angle=90, length=1.5, spacing=10),
    ]

    lsf_loads.set_path_effects(pe_list)

    ax.contourf(
        X_lsf,
        Y_lsf,
        lsf_func(X_lsf, Y_lsf),
        levels=[-1e10, 0.0, 1e10],
        colors=["#EAE9E6", "white"],
        zorder=10,
    )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.axis("off")

    return


# Plot the mesh
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6.64, 4.80))
show_fig_size_on_resize(fig)

create_mesh(ax=ax, lx=15.0, ly=10.0, nx=15, ny=10)

# Add text
texts = [
    [0.25, 0.25, "Clamped"],
    [0.74, 0.05, r"Void region $\bar{\Omega}$"],
    [0.61, 0.45, r"Analysis region $\Omega$"],
    [0.75, 0.8, r"Load"],
]

for t in texts:
    ax.text(*t, transform=ax.transAxes, fontsize=20, zorder=100)


plt.subplots_adjust(
    left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=0.0, hspace=0.0
)
fig.savefig("two_material_problem.jpg", dpi=1200)
plt.show()
