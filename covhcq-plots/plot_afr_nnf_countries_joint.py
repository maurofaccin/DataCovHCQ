#!/usr/bin/env python3
"""Plot results from non negative factorization."""

import json
from datetime import date

import cov_utils
import numpy as np
import pycountry
from matplotlib import cm, colors
from matplotlib import dates as mdates
from matplotlib import pyplot
from wordcloud import WordCloud


def sort_comps(matrix):
    """Sort components of the matrix for display."""
    smooth = np.apply_along_axis(np.convolve, 0, matrix, np.ones(21) / 21, mode="same")
    comps = np.argmax(smooth, axis=0)
    return np.argsort(comps)


def get_grid(n):
    """Compute the best grid for a given number of plots."""
    # it split in (ncols, mod, nrows)
    splits = [(*divmod(n, i + 1), i + 1) for i in range(n)]
    best = list(min(splits, key=lambda x: x[1] + np.abs(x[0] - x[2]) * 2))
    if best[1] > 0:
        if best[0] > best[2]:
            best[0] += 1
        else:
            best[2] += 1
    return sorted([best[0], best[2]], reverse=True)


def mix_color(col1, col2, weights: tuple = (0.5, 0.5)):
    """Take the average color."""
    c1 = colors.to_rgb(col1)
    c2 = colors.to_rgb(col2)
    weights = tuple(w / sum(weights) for w in weights)

    return colors.to_hex([cc1 * weights[0] + cc2 * weights[1] for cc1, cc2 in zip(c1, c2)])


def plot_wordcloud(
    figure: pyplot.Figure, components: dict, keymap: dict, bounds: list, conf: dict = {}
):
    """Print most important tags per component."""
    grid = get_grid(components["C"].shape[1])
    axes = figure.subplots(
        nrows=grid[0],
        ncols=grid[1],
        sharex=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={
            "left": bounds[0],
            "bottom": bounds[1],
            "right": bounds[2],
            "top": bounds[3],
            "hspace": 0.12,
            "wspace": 0.12,
        },
        subplot_kw={"frameon": True, "xticks": [], "yticks": []},
    )
    axes = axes.flatten()
    hts = {i: k for k, i in keymap["hashtags"].items()}

    if "title" in conf:
        figure.text(
            (bounds[0] + bounds[2]) / 2, 0.97, conf["title"], ha="center", fontsize="x-large"
        )

    for icomp, comp in enumerate(components["A"].T):
        label = f"{conf.get('prefix', 'X')}{icomp}"
        print("worldcoud", icomp, label)
        tags = [(hts[i], comp[i] / comp.max()) for i in np.argsort(comp)][::-1]

        color1 = f"C{icomp}"
        color2 = mix_color(color1, "#444444")
        wc = WordCloud(
            # contour_color="white",
            width=800,
            height=800,
            margin=0,
            background_color=None,
            colormap=colors.LinearSegmentedColormap.from_list("xx", [color1, color2]),
            mode="RGBA",
            max_words=500,
            min_font_size=8,
            relative_scaling=0,
            collocation_threshold=0,
        )
        wc.generate_from_frequencies(dict(tags))

        ax = axes[icomp]
        ax.imshow(wc, interpolation="antialiased")
        ax.set_facecolor(f"C{icomp}")
        ax.text(0, -10, label, color=f"C{icomp}", ha="right")
        for loc in ["top", "bottom", "right", "left"]:
            ax.spines[loc].set_color(f"C{icomp}")


def load_data(kind, ncomp, binary: bool = False):
    """Load data."""
    b = "_binary" if binary else ""
    matrices = np.load(f"../data/nntf{b}_{kind}/non_negative_parafac_rank_{ncomp}.npz")
    with open(f"../data/ht_cooccurrence_tensor_{kind.removesuffix('_norm')}.json", "rt") as fin:
        keys = json.load(fin)
    sorter = sort_comps(matrices["C"])

    a = matrices["A"][:, sorter]
    components = {
        "A": a,
        "B": matrices["B"][:, sorter],
        "C": matrices["C"][:, sorter],
        "S": matrices["C"][:, sorter] * a.sum(0),
    }
    for a, c in components.items():
        print(a, c.shape)
    return components, keys


def plot_time(figure: pyplot.Figure, components: dict, keymap: dict, bounds: list, conf: dict = {}):
    """Plot components along time in given axes."""
    ax = figure.subplots(
        gridspec_kw={
            "left": bounds[0],
            "bottom": bounds[1],
            "right": bounds[2],
            "top": bounds[3],
            "hspace": 0,
            "wspace": 0,
        },
    )
    try:
        xlabels = [date.fromisoformat(d) for d in keymap["days"]]
    except ValueError:
        xlabels = list(keymap["days"].keys())

    c0 = 0
    bars = []
    if isinstance(xlabels[0], str):
        pyplot.setp(ax.get_xticklabels(), rotation=90)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(xlabels) - 0.5)
        data = (components["S"] / components["S"].sum(1).reshape((-1, 1))).T
        sorter = np.argsort(data[conf.get("sorters", []), :].sum(0))
        xlabels = [xlabels[s] for s in sorter]
        data = data[:, sorter]
    else:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator, formats=["%b\n%Y", "%b", "", "", "", ""])
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim(date.fromisoformat("2020-01-01"), date.fromisoformat("2021-09-30"))
        data = components["S"].T

    for icomp, comp in enumerate(data):
        label = f"{conf.get('prefix', 'X')}{icomp}"
        if isinstance(xlabels[0], str):
            bar = ax.bar(xlabels, comp, 1.05, bottom=c0, label=label, lw=0)
            for it, text in enumerate(xlabels):
                ax.text(
                    it,
                    0.01,
                    pycountry.countries.get(alpha_2=text).name,
                    color="#cccccc",
                    va="bottom",
                    ha="center",
                    rotation=90,
                    alpha=0.3,
                    fontsize="x-small",
                )
        else:
            comp = cov_utils.smooth_convolve(comp, 7)
            bar = ax.fill_between(xlabels, c0, c0 + comp, label=label)
        bars.append(bar)
        c0 += comp

    if isinstance(xlabels[0], str):
        ax.set_ylabel("Normalized Component Strenght")
    else:
        ax.set_ylabel("Component Strenght")
        ax.legend()

    ax.set_yticks([])


def plot_time_both(figure: pyplot.Figure, components: dict, keymap: dict, bounds: list):
    """Plot components along time in given axes."""
    axes = figure.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={
            "left": bounds[0],
            "bottom": bounds[1],
            "right": bounds[2],
            "top": bounds[3],
            "hspace": 0,
            "wspace": 0,
        },
    )
    try:
        xlabels = [date.fromisoformat(d) for d in keymap["days"]]
    except ValueError:
        xlabels = list(keymap["days"].keys())

    ax = axes[0]
    ncomponents = components["C"].shape[1]
    ax.pcolormesh(
        xlabels,
        range(ncomponents),
        (components["C"] / components["C"].sum(0).reshape((1, -1))).T,
        cmap="hot_r",
        norm=colors.SymLogNorm(1, 0.01),
    )
    ax.set_yticks(range(components["C"].shape[1]))
    ax.set_yticklabels([f"C{i}" for i in range(components["C"].shape[1])])
    ax.yaxis.set_tick_params(which="major", length=15)
    ax.xaxis.set_tick_params(which="major", length=0)

    ax = axes[1]
    c0 = 0
    bars = []
    # for icomp, comp in enumerate((components["C"] / components["C"].sum(1).reshape((-1, 1))).T):
    if isinstance(xlabels[0], str):
        pyplot.setp(ax.get_xticklabels(), rotation=90)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(xlabels) - 0.5)
        data = (components["C"] / components["C"].sum(1).reshape((-1, 1))).T
    else:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator, formats=["%b\n%Y", "%b", "", "", "", ""])
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim(date.fromisoformat("2020-01-01"), date.fromisoformat("2021-09-30"))
        data = components["C"].T

    for icomp, comp in enumerate(data):
        if isinstance(xlabels[0], str):
            bar = ax.bar(xlabels, comp, 1.05, bottom=c0, label=f"C{icomp}", lw=0)
        else:
            # comp = np.convolve(np.ones(7) / 5, comp, mode="same")
            comp = cov_utils.smooth_convolve(comp, 7)
            bar = ax.fill_between(xlabels, c0, c0 + comp)
        bars.append(bar)
        c0 += comp
    ax.set_ylabel("frequency")

    ax.set_yticks([])

    cax = figure.add_axes(
        [bounds[0] - 0.02, (bounds[1] + bounds[3]) / 2, 0.02, (bounds[3] - bounds[1]) / 2]
    )
    cbar = figure.colorbar(
        cm.ScalarMappable(cmap=colors.ListedColormap([f"C{i}" for i in range(ncomponents)])),
        cax=cax,
    )
    cbar.set_ticks([])

    # figure.align_labels(axs=axes)


def main(ncomp):
    """Do the main."""
    kind = ["geo_early_norm", "geo_late_norm"]
    kind = ["geo_early", "geo_late"]
    kind = ["full", "nofr"]
    kind = ["hcq"]
    config = {
        "geo_early": {"title": "First period", "sorters": [1, 2, 4], "prefix": "C"},
        "geo_late": {"title": "Second Period", "sorters": [1, 3], "prefix": "D"},
        "geo_early_norm": {"title": "First period"},
        "geo_late_norm": {"title": "Second Period"},
        "full": {"title": "Full", "prefix": "A"},
        "nofr": {"title": "No France", "prefix": "B"},
        "hcq": {"title": "HCQ", "prefix": "E"},
    }

    figure = pyplot.figure(figsize=(5 * len(kind), 7.5))

    vsep = 0.35
    seps = [0.05, 0.05, 0.02, 0.05]

    for ik, k in enumerate(kind):
        hmin, hmax = ik / len(kind), (ik + 1) / len(kind)
        components, keymap = load_data(k, ncomp, binary=False)
        bounds = [hmin + seps[0], seps[1], hmax - seps[2], vsep]
        plot_time(figure, components, keymap, bounds, conf=config[k])
        bounds = [hmin + seps[0], vsep + 0.02, hmax - seps[2], 1 - seps[3]]
        plot_wordcloud(figure, components, keymap, bounds, conf=config[k])

    pyplot.savefig(f"plot_afr_nnf_countries_{'-'.join(kind)}.pdf", dpi=300)


if __name__ == "__main__":
    main(9)
