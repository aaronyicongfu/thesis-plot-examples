import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

import scienceplots

plt.style.use(["science"])


def annotate_slope(
    ax, pt0, pt1, slide=0.05, scale=0.9, hoffset=0.0, voffset=-0.1, voffset_text=-0.35
):
    """
    Annotate the slope on a log-log plot

    Args:
        ax: Axes
        pt0, pt1: tuple of (x, y) where x and y are original data (not exponent)
    """

    x0, y0 = pt0
    x1, y1 = pt1

    # Make sure pt0 is always the lower one
    if y0 > y1:
        (x0, y0), (x1, y1) = (x1, y1), (x0, y0)

    dy = np.log10(y1) - np.log10(y0)
    dx = np.log10(x1) - np.log10(x0)
    slope = dy / dx

    x0 *= 10.0**hoffset
    y0 *= 10.0**voffset

    x0 = 10.0 ** (np.log10(x0) + dx * slide)
    y0 = 10.0 ** (np.log10(y0) + dy * slide)

    x1 = 10.0 ** (np.log10(x0) + dx * scale)
    y1 = 10.0 ** (np.log10(y0) + dy * scale)

    # Create a right triangle using Polygon patch
    triangle = patches.Polygon(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
        ],
        closed=True,
        # fill=False,
        edgecolor="black",
        facecolor="gray",
        zorder=100,
        lw=0.5,
    )

    # Add the triangle patch to the plot
    ax.add_patch(triangle)

    # Annotate the slope
    ax.annotate(
        f"{slope:.2f}",
        xy=(np.sqrt(x0 * x1), y0 * 10.0**voffset_text),
        verticalalignment="baseline",
        horizontalalignment="center",
    )

    return


def plot_precision(df):
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(5.3, 4.0),
        constrained_layout=True,
    )

    # Get ggplot colors
    colors = plt.style.library["ggplot"]["axes.prop_cycle"].by_key()["color"]
    for i, (Np_1d, sub_df) in enumerate(df.groupby("Np_1d")):
        # Get averaged slope
        x = sub_df["h"]
        y = sub_df["stress_norm"]
        slope, _ = np.polyfit(np.log10(x), np.log10(y), deg=1)
        label = f"$p={Np_1d - 1}, \Delta:{slope:.2f}$"
        ax.loglog(
            x,
            y,
            "-o",
            label=label,
            lw=1.0,
            markeredgewidth=1.0,
            markersize=6.0,
            markeredgecolor="black",
            color=colors[i],
        )

    ymin, ymax = ax.get_ylim()
    v_off = -np.log10(ymax / ymin) * 0.02
    v_off_txt = -np.log10(ymax / ymin) * 0.035

    for Np_1d, sub_df in df.groupby("Np_1d"):
        x = sub_df["h"]
        y = sub_df["stress_norm"]
        x0, x1 = x.iloc[-2:]
        y0, y1 = y.iloc[-2:]
        annotate_slope(
            ax,
            (x0, y0),
            (x1, y1),
            voffset=v_off,
            voffset_text=v_off_txt,
        )

        ylabel = r"$\left[\int_h  \text{tr}((\mathbf{S} - \mathbf{S}_h)^T(\mathbf{S} - \mathbf{S}_h)) d\Omega\right]^{1/2}$"

        ax.set_ylim(bottom=ymin * 10.0 ** (v_off_txt * 1.05))
        ax.legend()
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(ylabel)

    return fig, ax


if __name__ == "__main__":
    df = pd.read_csv("line_plot_example.csv")
    fig, ax = plot_precision(df)

    fig.savefig("line_plot_example.pdf")
    fig.savefig("line_plot_example.svg")

    plt.show()
