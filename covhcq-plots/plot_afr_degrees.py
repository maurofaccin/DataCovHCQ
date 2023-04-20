#!/usr/bin/env python3

from datetime import date

import numpy as np
from matplotlib import dates as mdates
from matplotlib import pyplot, ticker

import cov_afr
import cov_utils


def main():
    """Do the main."""
    data = cov_utils.load_csv("../data/inward_degree.csv", transpose=True)
    data_nofr = cov_utils.load_csv("../data/inward_degree_noFR.csv", transpose=True)
    data = {k: np.ma.masked_less_equal(v, 0) for k, v in data.items()}
    dates = [date.fromisoformat(d + "-01") for d in cov_afr.cycle_months()]
    print(*data.keys(), sep="\n")

    fig, ax = pyplot.subplots(
        1,
        1,
        sharex=True,
        figsize=(6, 3),
        gridspec_kw={"hspace": 0, "top": 0.95, "bottom": 0.15, "right": 0.98},
    )
    cmap = pyplot.cm.get_cmap("tab20")

    ax.plot(dates, data["full_inward"], label="FULL", color=cmap(0))
    ax.plot(dates, data_nofr["full_inward"], label="FULL noFR", color=cmap(1))
    ax.plot(dates, data["hydchl_inward"], label="HCQ", color=cmap(2))
    ax.plot(dates, data_nofr["hydchl_inward"], label="HCQ noFR", color=cmap(3))
    ax.set_ylim(0.65, 1.0)
    # ax.set_yticks([0.8, 0.9, 1.0])
    ax.legend(
        loc="lower center",
        ncol=2,
        frameon=True,
        labelspacing=0.2,
        handlelength=1.5,
        handleheight=0.7,
        columnspacing=1,
    )
    ax.grid()
    ax.set_ylabel("total trappig probability")

    # ax = axes[1]
    # ax.plot(dates, data["full_num_edges"], label="FULL noFR", color=cmap(0))
    # ax.plot(dates, data["hydchl_num_edges"], label="HYD noFR", color=cmap(2))
    # ax.set_ylabel("tweet count")
    # ax.set_yticks(range(0, 1000001, 300000))
    # ax.yaxis.set_major_formatter(lambda x, y: f"{x/1000:.0f}k" if x > 0 else "0")
    # ax.grid()

    # for nn, ax in enumerate(axes):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, formats=["%b\n%Y", "%b", "", "", "", ""])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    # fig.align_labels()
    # pyplot.tight_layout(h_pad=0)
    # pyplot.show()
    pyplot.savefig("plot_afr_degrees.pdf")


if __name__ == "__main__":
    main()
