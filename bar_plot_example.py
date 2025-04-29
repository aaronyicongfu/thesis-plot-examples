import numpy as np
import pandas as pd
import argparse
from time import time
import subprocess
from os.path import join
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import scienceplots

# # Get ggplot colors
colors = plt.style.library["ggplot"]["axes.prop_cycle"].by_key()["color"]

plt.style.use(["science"])

from typing import List


def plot_cpu_time_breakdown(df):
    num_Np_1d = len(set(df["Np_1d"]))
    fig, axs = plt.subplots(
        ncols=num_Np_1d,
        nrows=1,
        figsize=(5.3 * num_Np_1d, 4.0),
        constrained_layout=True,
    )

    yname_label_map = {
        "jacobian_time": "Jacobian assembly",
        "residual_time": "Residual assembly",
        "chol_init_time": "Cholesky initialization",
        "chol_factor_time": "Cholesky factorization",
        "chol_solve_time": "Cholesky solve",
        "other": "Other",
    }

    # Derived column
    df["other"] = df["sol_time"]
    for yname in yname_label_map.keys():
        if yname != "other":
            df["other"] -= df[yname]

    for i, (Np_1d, sub_df) in enumerate(df.groupby("Np_1d")):

        # Determine the bar width
        bar_width = (
            0.8
            * (np.log10(max(sub_df["h"])) - np.log10(min(sub_df["h"])))
            / len(sub_df["h"])
            / len(yname_label_map)
        ) * np.log(10)

        for j, (yname, label) in enumerate(yname_label_map.items()):
            axs[i].bar(
                sub_df["h"]
                + (j - (len(yname_label_map) - 1) / 2) * bar_width * sub_df["h"],
                sub_df[yname],
                bar_width
                * sub_df["h"],  # transformation from linear space to log space
                label=label if i == 0 else None,
                facecolor=colors[j],
                edgecolor="black",
                linewidth=0.5,
                alpha=1.0,
            )

        axs[i].set_xscale("log")
        axs[i].set_yscale("log")

        # Remove all existing ticks
        axs[i].tick_params(axis="x", which="both", length=0, labelbottom=False)

        # Set new ticks with explicit positions and labels
        axs[i].set_xticks(sub_df["h"])
        axs[i].set_xticklabels(
            sub_df["h"].apply(lambda x: f"{x:.1e}"), rotation=45, ha="right"
        )

        # Make ticks point outward/downward
        axs[i].tick_params(
            axis="x", which="major", direction="out", length=3, labelbottom=True
        )

        # Remove the ticks from top, if any
        axs[i].tick_params(top=False)

        axs[i].set_xlabel(r"$h$")
        axs[i].set_ylabel(r"CPU time (s)")
        axs[i].set_title(f"$p={Np_1d - 1}$")

    fig.legend(
        loc="upper center",
        ncols=6,
        bbox_to_anchor=(0.5, 1.07),
        bbox_transform=fig.transFigure,
    )

    return fig, axs


if __name__ == "__main__":
    df = pd.read_csv("bar_plot_example.csv")
    fig, ax = plot_cpu_time_breakdown(df)

    fig.savefig("bar_plot_example.jpg", dpi=1200)

    plt.show()
