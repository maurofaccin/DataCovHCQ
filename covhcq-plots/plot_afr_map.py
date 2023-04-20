#!/usr/bin/env python3
"""Plot connection between countries in Africa."""

import colorsys
import csv
import gzip
import json
import pathlib

import cov_afr
import matplotlib
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
from adjustText import adjust_text
from cartopy import crs as ccrs
from cartopy import feature
from matplotlib import artist, colors, pyplot
from plot_afr_nnf_countries_joint import load_data as load_comp_strength
from scipy import sparse

DATAPATH = "../data/african_tweets_{}_"
CLIP = 70


class FilteredArtistList(artist.Artist):
    """A simple container to filter multiple artists at once."""

    def __init__(self, artist_list, filter):
        super().__init__()
        self._artist_list = artist_list
        self._filter = filter

    def draw(self, renderer):
        renderer.start_rasterizing()
        renderer.start_filter()
        for a in self._artist_list:
            a.draw(renderer)
        renderer.stop_filter(self._filter)
        renderer.stop_rasterizing()


class BaseFilter:
    def get_pad(self, dpi):
        return 0

    def process_image(padded_src, dpi):
        raise NotImplementedError("Should be overridden by subclasses")

    def __call__(self, im, dpi):
        pad = self.get_pad(dpi)
        padded_src = np.pad(im, [(pad, pad), (pad, pad), (0, 0)], "constant")
        tgt_image = self.process_image(padded_src, dpi)
        return tgt_image, -pad, -pad


class GaussianFilter(BaseFilter):
    """Simple Gaussian filter."""

    def __init__(self, sigma, alpha=0.5, color=(0, 0, 0)):
        self.sigma = sigma
        self.alpha = alpha
        self.color = color

    def get_pad(self, dpi):
        return int(self.sigma * 3 / 72 * dpi)

    def process_image(self, padded_src, dpi):
        tgt_image = np.empty_like(padded_src)
        tgt_image[:, :, :3] = self.color
        tgt_image[:, :, 3] = smooth2d(padded_src[:, :, 3] * self.alpha, self.sigma / 72 * dpi)
        return tgt_image


def smooth2d(A, sigma=3):
    window_len = max(int(sigma), 3) * 2 + 1
    A = np.apply_along_axis(smooth1d, 0, A, window_len)
    A = np.apply_along_axis(smooth1d, 1, A, window_len)
    return A


def smooth1d(x, window_len):
    # copied from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1 : -window_len + 1]


class DropShadowFilter(BaseFilter):
    def __init__(self, sigma, alpha=0.3, color=(0, 0, 0), offsets=(0, 0)):
        self.gauss_filter = GaussianFilter(sigma, alpha, color)
        self.offset_filter = OffsetFilter(offsets)

    def get_pad(self, dpi):
        return max(self.gauss_filter.get_pad(dpi), self.offset_filter.get_pad(dpi))

    def process_image(self, padded_src, dpi):
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        t2 = self.offset_filter.process_image(t1, dpi)
        return t2


def hex_to_rgb(hexcolor):
    return (
        int(hexcolor[1:3], 16) / 255,
        int(hexcolor[3:5], 16) / 255,
        int(hexcolor[5:7], 16) / 255,
    )


def rgb_to_hex(rgb):
    _hex = "#"
    for x in rgb:
        _hex += str(hex(int(x * 255)))[2:]
    return _hex


def darken(hexcolor, step=0.1):
    if isinstance(hexcolor, str):
        rgb = hex_to_rgb(hexcolor)
    else:
        rgb = hexcolor

    hls = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(hls[0], max(0, hls[1] - step), hls[2])


def noverlap(pos, node_size):
    rgn = np.random.default_rng()
    nodes = list(pos.keys())

    for i in range(100000):
        n1, n2 = rgn.choice(nodes, size=2, replace=False)
        pos[n1], pos[n2] = move_nodes(pos[n1], pos[n2], node_size[n1], node_size[n2])
    return pos


def move_nodes(p1, p2, s1, s2, weight=[1, 1]):
    p1 = np.array(p1)
    p2 = np.array(p2)

    dist = np.linalg.norm(p2 - p1)
    overlap = s1 + s2 - dist

    if overlap < 0:
        return p1, p2

    uvec = (p2 - p1) / dist
    weight = np.array(weight)

    p1new = p1 - uvec * overlap * s2 / (s1 + s2)
    p2new = p2 + uvec * overlap * s1 / (s1 + s2)
    return p1new, p2new


def datapath(kind, data):
    """Return the path tu the file."""
    path = pathlib.Path(DATAPATH.format(kind))
    return path.parent / (path.name + data)


def find_connections(kind: str = "full", tau: int = -1) -> sparse.csr_matrix:
    """Find connection between countries."""
    head = sparse.load_npz(datapath(kind, "head.npz"))
    tail = sparse.load_npz(datapath(kind, "tail.npz"))

    if True:
        adj, new_umap, other_components = cov_afr.adjacency(
            ({i: i for i in range(tail.shape[1])}, tail.T.asfptype(), head.T.asfptype()),
            tau=tau,
            fix_sources="symmetrize",
        )
    else:
        weight = np.array(head.sum(1), dtype=np.float64).flatten()
        weight[weight == 0] = 1

        adj = tail.T @ sparse.diags(weight**tau) @ head
        # sum on the first dym (rows) to get indegree
        # sum on the second dimension (columns) to get outdegree
        new_umap = {i: i for i in range(adj.shape[0])}
    return adj, new_umap


def country_projector(kind: str = "full", ubunch: set = set()):
    """Read user map and return a projector."""
    with open(datapath(kind, "userid.tsv"), "rt") as fin:
        reader = csv.DictReader(fin, dialect=csv.excel_tab)
        if len(ubunch) == 0:
            user_map = {r["matrix_id"]: r["from_user_id"] for r in reader}
        else:
            user_map = {
                r["matrix_id"]: r["from_user_id"] for r in reader if int(r["matrix_id"]) in ubunch
            }
    print("N users", len(user_map))

    with gzip.open("../data/african_tweets_users_extended.csv.gz", "rt") as fin:
        reader = csv.DictReader(fin, dialect=csv.excel)
        user_country_map = {
            row["from_user_id"]: row["geo_coding_extened"]
            if len(row["geo_coding_extened"]) <= 2
            else ""
            for row in reader
        }
        country_map = {c: i for i, c in enumerate(sorted(set(user_country_map.values())))}

    proj = sparse.csr_matrix(
        (
            np.ones(len(user_map)),
            (
                np.arange(len(user_map)),
                [country_map[user_country_map.get(uid, "")] for uid in user_map.values()],
            ),
        ),
        shape=(len(user_map), len(country_map)),
    )

    return proj, country_map


def plot_graph(graph: nx.DiGraph, kind: str = "full"):
    """Plot the network on the globe."""
    extent = [-CLIP - 5, CLIP + 5, -40, 70]
    crs = ccrs.EqualEarth()

    extent_crs = crs.transform_points(
        ccrs.PlateCarree(), np.array([extent[0], extent[1]]), np.array([extent[2], extent[3]])
    )
    figsize = np.asarray([extent_crs[1, 0] - extent_crs[0, 0], extent_crs[1, 1] - extent_crs[0, 1]])
    figsize = 7 * figsize[::-1] / figsize.max()
    fig = pyplot.figure(figsize=figsize)
    skip = 0.01
    ax = fig.add_axes([skip, skip, 1 - 2 * skip, 1 - 2 * skip], projection=crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.axis("off")

    # add first features
    bg_color = "#D4EEEF"
    bg_color = "#D6D6D6"

    land_50m = feature.NaturalEarthFeature(
        "physical",
        "land",
        "50m",
        edgecolor=darken(bg_color, step=0.3),
        linewidth=0.5,
        facecolor=f"{bg_color}",
    )
    land_feat = ax.add_feature(land_50m, zorder=-1)

    gauss = GaussianFilter(20, alpha=0.4)
    shadow = FilteredArtistList([land_feat], gauss)
    shadow.set_zorder(-3)
    ax.add_artist(shadow)

    countries_50m = feature.NaturalEarthFeature(
        "cultural",
        "admin_0_boundary_lines_land",
        "50m",
        edgecolor=darken(bg_color, step=0.2),
        linewidth=0.3,
        facecolor="none",
    )
    ax.add_feature(countries_50m, zorder=-1)

    # remove non-geolocalized nodes
    graph.remove_node("")

    with open("../data/countries.json", "rt") as fin:
        pos = {
            c["cca2"]: crs.transform_point(
                *np.clip(c["latlng"][::-1], -CLIP, CLIP), ccrs.PlateCarree()
            )
            for c in json.load(fin)
            if c["cca2"] in graph.nodes()
        }

    if "" in pos:
        pos[""] = (-10, -1)

    density = np.array([v for _, v in graph.nodes(data="density")])
    users = np.array([v for _, v in graph.nodes(data="users")])

    node_radius = np.sqrt(users)
    node_radius = 1000000 * node_radius / node_radius.max()

    pos = noverlap(pos, {n: s for n, s in zip(graph.nodes(), node_radius)})

    edges = [e for e in graph.edges(data="weight") if e[2] > 0 and e[0] != e[1]]
    ew = np.max([e[2] for e in edges])
    print(ew)
    # set a fixed number to be able to compare
    ew = 1e-4
    edict = {(e[0], e[1]): e[2] for e in edges}

    for (n0, n1), w in edict.items():
        x0, y0 = pos[n0]
        x1, y1 = pos[n1]
        pyplot.annotate(
            "",
            (x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": f"wedge,tail_width={w / (ew + 0)}",
                "connectionstyle": "arc3,rad=0.3",
                "facecolor": "#888888",
                "lw": 0,
                "alpha": 0.8,
            },
            zorder=0,
        )

    cmap = pyplot.get_cmap("YlOrRd")

    components, keys = load_comp_strength("geo_early", 9)
    print(components["S"])
    strength = components["S"].T / components["S"].sum(1)
    strength = strength[[2, 4], :].sum(0)

    norm = 0.25

    nodes = []
    for node, radius, dens in zip(graph.nodes(), node_radius, density):
        print(node, radius, dens)
        dens = strength[keys["days"][node]] / norm
        circ = pyplot.Circle((pos[node]), radius=radius, facecolor=cmap(dens))
        ax.add_patch(circ)
        nodes.append(circ)

    text = [
        pyplot.text(
            position[0],
            position[1],
            country,
            path_effects=[pe.Stroke(linewidth=1, foreground="w", alpha=0.5), pe.Normal()],
            fontsize=8 + 20 * graph.nodes[country]["users"] / users.max(),
            va="center",
            ha="center",
        )
        for country, position in pos.items()
    ]
    adjust_text(text)

    # add legend
    cax = ax.inset_axes([0.55, 0.03, 0.4, 0.03])
    pyplot.colorbar(
        matplotlib.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=norm), cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )
    cax.set_title("Relative HCQ component strength")

    # pyplot.tight_layout()
    pyplot.savefig(f"plot_afr_map_{kind}.pdf")


def normalize(matrix):
    n = np.outer(matrix.sum(1), matrix.sum(0)) / matrix.sum()
    # n[n == 0] = 1
    return n


def build_graph(kind="full", tau=-1):
    """Build the graph."""
    fname = pathlib.Path(f"../data/map_graph_{kind}_tau_{tau}.graphml")
    fname.parent.mkdir(exist_ok=True)
    if fname.is_file():
        print("reading the file", fname)
        return nx.read_graphml(fname)

    adj, new_umap = find_connections(kind=kind, tau=tau)
    proj, country_map = country_projector(kind=kind, ubunch=set(new_umap.values()))

    adj = (proj.T @ adj @ proj).toarray()

    nusers = {c[1]: n for c, n in np.ndenumerate(proj.sum(0))}
    density = {c[0]: d for c, d in np.ndenumerate(adj.diagonal() / adj.sum(0))}

    # TODO: keep only more-than-expected links
    adj[adj < normalize(adj)] = 0
    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    # add attributes
    nx.set_node_attributes(graph, nusers, name="users")
    nx.set_node_attributes(graph, density, name="density")
    graph = nx.relabel_nodes(graph, {v: k for k, v in country_map.items()})

    # save to disk
    nx.write_graphml_lxml(graph, fname)
    return graph


def main(kind):
    """Do the main."""
    graph = build_graph(kind=kind, tau=0)
    plot_graph(graph, kind=kind)


if __name__ == "__main__":
    main("full")
    # main("hydchl")
