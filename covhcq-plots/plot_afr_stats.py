#!/usr/bin/env python3
"""Print some statistics."""


import csv
from datetime import date

import cov_utils
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot, ticker
from scipy import sparse


def get_data(kind):
    """Retrieve data."""
    tail = sparse.load_npz(f"../data/african_tweets_{kind}_tail.npz")
    head = sparse.load_npz(f"../data/african_tweets_{kind}_head.npz")

    data = {
        "num_tweets": tail.shape[0],
        "num_retweets": int(head.sum()),
        "users_involved": tail.shape[1],
    }

    marginal = tail.sum(0)
    data["users_tweeting"] = int((marginal > 0).sum())
    marginal = head.sum(0)
    data["users_retweeting"] = int((marginal > 0).sum())

    return data


def label_fmt(x, pos):
    """Human readable format."""
    if x == 0:
        return "0"
    if x % 1000000 == 0:
        return f"{int(x//100000)}M"
    if x % 1000 == 0:
        return f"{x//1000:.0f}k"
    return f"{x}"


def plot_lockdown(ax, text=False):
    """Add lockdown to axis."""
    with open("../data/lockdowns.tsv", "rt") as fin:
        reader = csv.DictReader(fin, dialect=csv.excel_tab)
        for lockdown in reader:
            if text:
                ax.text(
                    date.fromisoformat(lockdown["start"]),
                    ax.get_ylim()[1] * 1.08,
                    f"Conf {lockdown['confinement'][-1]}",
                    color="#666666",
                )
            ax.axvspan(
                date.fromisoformat(lockdown["start"]),
                date.fromisoformat(lockdown["end"]),
                alpha=0.2,
                color="grey",
            )


def plot_time(axes):
    """Plot temporal stats."""
    # load tweet counts
    pdata = pd.read_csv("../data/african_tweets_counts.csv.gz", index_col=0, parse_dates=True)
    # load incidence counts
    cases = cov_utils.load_cases("FR", transpose=True)

    # save tweeets and retweets for covid and hcq
    covid_count = pdata[
        ["_".join([pre, post]) for pre in ["tweets", "retweets"] for post in ["covid", "both"]]
    ].sum(axis=1)
    hcq_count = pdata[
        [
            "_".join([pre, post])
            for pre in ["tweets", "retweets"]
            for post in ["hydroxychloroquine", "both"]
        ]
    ].sum(axis=1)

    # plot cases
    ax = axes[0]
    yes = cases["New_cases"]
    xes = cases["day"]
    yes = [x if (x is not None and not isinstance(x, str)) else 0 for x in yes]
    inc = ax.fill_between(
        xes,
        cov_utils.smooth_convolve(yes, 7),
        color="#999999",
        alpha=0.5,
        linewidth=0,
        label="Incidence",
    )
    ax.set_ylabel("Cases (FR)")
    ax.yaxis.set_label_coords(-0.1, 0.5)

    print("HCQ", hcq_count.sum())
    ax = axes[2]
    (hcq,) = ax.plot(hcq_count.index, hcq_count, color="C1", label="HCQ")

    print("COV", covid_count.sum())
    ax = axes[1]
    (cov,) = ax.plot(covid_count.index, covid_count, color="C0", label="Covid")
    ax.set_ylabel("Tweet frequency")
    ax.yaxis.set_label_coords(-0.1, 0.0)
    ax.legend(handles=[inc, cov, hcq])

    for ax in axes:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(label_fmt))

    # Events
    events = [
        date.fromisoformat(x)
        for x in [
            "2020-03-27",
            "2020-05-22",
            "2020-06-05",
            # "2020-06-17",
            "2020-06-24",
        ]
    ]
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    ax2.set_xticks(
        events,
        labels=[e.strftime("%d %b") for e in events],
        fontsize="x-small",
        rotation=45,
        ha="left",
    )
    for event in events:
        for ax in axes:
            ax.axvline(event, color="#aaaaaa", zorder=-10)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator, formats=["%b\n%Y", "%b", "", "", "", ""])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def plot_country_freq(ax):
    """Add plot of country freq."""
    users = pd.read_csv("../data/african_users_extended.csv.gz")
    # only users with tags
    users = users[users.geo_coding_extened.str.len() == 2]
    count = users.geo_coding_extened.value_counts()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(label_fmt))

    count.plot.bar(
        ax=ax,
        width=1.1,
        color=["xkcd:green" if i % 2 == 0 else "xkcd:grass green" for i in range(len(count))],
    )
    print(count)

    ax.semilogy()
    # ax.set_ylim(100, 500000)
    ax.set_xlabel("Countries")
    ax.set_ylabel("User count")
    ax.yaxis.set_label_coords(-0.1, 0.5)


def main():
    """Do the main."""
    figure = pyplot.figure(figsize=(6, 5))
    axes = figure.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        gridspec_kw={
            "hspace": 0,
            "wspace": 0,
            "left": 0.15,
            "top": 0.9,
            "right": 0.98,
            "bottom": 0.4,
            "height_ratios": [0.2, 0.4, 0.4],
        },
    )
    plot_time(axes)

    ax = figure.add_axes([0.15, 0.1, 0.83, 0.2])
    plot_country_freq(ax)

    figure.align_ylabels(axs=[axes[1], ax])

    pyplot.savefig("plot_afr_stats.pdf")


if __name__ == "__main__":
    main()
