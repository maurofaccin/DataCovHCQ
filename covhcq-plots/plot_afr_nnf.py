#!/usr/bin/env python3
"""Plot results from non negative factorization."""

import json
import pathlib
from datetime import date

import numpy as np
from matplotlib import colors
from matplotlib import dates as mdates
from matplotlib import pyplot
from wordcloud import WordCloud

import cov_afr
import cov_utils

pyplot.style.use("mplrc")


def load_hts(kind):
    """Load index to hashtag map."""
    with open(f"../data_afr/ht_cooccurrence_tensor_{kind}.json", "rt") as fin:
        hts = json.load(fin)
    return {int(v): k for k, v in hts["hashtags"].items()}, {
        int(v): k for k, v in hts["days"].items()
    }
    # with open("../data_afr/ht_cooccurrence_tensor.json", "rt") as fin:
    #     # load key to index mapping as in tensor
    #     mapping = {int(v): k for k, v in json.load(fin)["hashtags"].items()}
    # return mapping


def get_grid(n):
    # it split in (ncols, mod, nrows)
    splits = [(*divmod(n, i + 1), i + 1) for i in range(n)]
    best = list(min(splits, key=lambda x: x[1] + np.abs(x[0] - x[2]) * 2))
    if best[1] > 0:
        if best[0] > best[2]:
            best[0] += 1
        else:
            best[2] += 1
    return sorted([best[0], best[2]], reverse=False)


def print_components(hts: dict, mat: np.ndarray, filename=None):
    """Print most important tags per component."""

    ncomps = mat.shape[1]
    nc, nr = get_grid(ncomps)
    print(ncomps, nc, nr)
    fig, axes = pyplot.subplots(
        ncols=nc,
        nrows=nr,
        figsize=(nc * 5 / max(nc, nr), nr * 5 / max(nc, nr)),
        subplot_kw={"frame_on": False, "xticks": [], "yticks": []},
    )
    axes = axes.flatten()

    for icomp, comp in enumerate(mat.T):
        tags = [(hts[i], comp[i]) for i in np.argsort(comp)][::-1]
        wc = WordCloud(
            width=800, height=800, margin=0, background_color=None, colormap="cool", mode="RGBA"
        )
        wc.generate_from_frequencies(dict(tags))
        axes[icomp].imshow(wc, interpolation="antialiased")
        axes[icomp].text(1, 1.1, f"C{icomp}")

    pyplot.tight_layout(pad=0.2)
    if filename is None:
        filename = f"plot_afr_nnf_{ncomps}_words.svg"
    pyplot.savefig(filename, dpi=300)


def sort_comps(matrix):
    """Sort components of the matrix for display."""
    smooth = np.apply_along_axis(np.convolve, 0, matrix, np.ones(21) / 21, mode="same")
    comps = np.argmax(smooth, axis=0)
    return np.argsort(comps)


def main(COMP=8):
    """Do the main."""
    kind = "nofr"
    mats = np.load(f"../data_afr/nntf_{kind}/non_negative_parafac_rank_{COMP}.npz")

    # create outfolder
    newfolder = pathlib.Path(f"./plot_afr_nnf_{kind}/")
    newfolder.mkdir(exist_ok=True)

    # dates
    dates = [date.fromisoformat(d + "-01") for d in cov_afr.cycle_months()]
    dates = cov_utils.daterange("2020-01-01", "2021-10-01")
    hash_map, days_map = load_hts(kind)
    sorter = sort_comps(mats["C"])
    comps = {
        "A": mats["A"][:, sorter],
        "B": mats["B"][:, sorter],
        "C": mats["C"][:, sorter],
    }

    print_components(hash_map, comps["B"], filename=newfolder / f"plot_afr_nnf_{COMP}_words.svg")

    fig, axes = pyplot.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0})

    ax = axes[0]
    ax.tick_params(axis="x", bottom=False)
    ax.pcolormesh(
        list(days_map.keys()),
        range(COMP),
        comps["C"].T,
        cmap="hot_r",
        norm=colors.SymLogNorm(1, 0.01),
    )
    ax.set_ylabel("components")

    ax = axes[1]
    c0 = 0
    for icomp, comp in enumerate(comps["C"].T):
        c1 = c0 + comp
        ax.fill_between(list(days_map.keys()), c0, c1, label=f"C{icomp}", lw=2)
        c0 = c1
    ax.legend(ncol=2)
    ax.set_ylabel("frequency")

    for nn, ax in enumerate(axes):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator, formats=["%b\n%Y", "%b", "", "", "", ""])
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    fig.align_labels()

    pyplot.savefig(newfolder / f"plot_afr_nnf_{COMP}.svg")
    pyplot.close()


if __name__ == "__main__":
    for c in range(5, 18):
        main(c)
