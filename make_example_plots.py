"""Generate the paper conceptual figure: example_plots.pdf."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


SEED = 0
N = 2000
EPS = 0.5


def _resolve_paper_plots_dir() -> Path:
    out_dir = Path("Plots_paper").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_graph(ax, edges, title: str) -> None:
    g = nx.DiGraph(edges)
    nodes = set(g.nodes())

    if nodes == {"M", "G", "S"} and ("S", "M") in g.edges() and ("S", "G") in g.edges():
        pos = {"M": (-1, 0), "S": (0, 0), "G": (1, 0)}
    elif nodes == {"M", "G", "S"}:
        pos = {"M": (-1, 0), "G": (0, 0), "S": (1, 0)}
    else:
        pos = {"M": (-1, 0), "G": (0, 0), "S": (1, 0), "E": (0, 1)}

    pos = {node: pos[node] for node in g.nodes() if node in pos}
    nx.draw(
        g,
        pos,
        ax=ax,
        with_labels=True,
        node_size=2000,
        node_color="lightgray",
        arrowsize=20,
        font_size=12,
    )

    if "E" in pos:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.65)
        ax.text(0, 1.9, title, ha="center", va="top", fontsize=16)
    else:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.8, 0.55)
        ax.text(0, 0.56, title, ha="center", va="top", fontsize=16)

    ax.axis("off")


def _residuals(x: np.ndarray, given: np.ndarray) -> np.ndarray:
    beta = np.polyfit(given, x, 1)
    return x - (beta[0] * given + beta[1])


def main() -> None:
    np.random.seed(SEED)

    # 1) Chain: M -> G -> S
    m1 = np.random.normal(size=N)
    g1 = m1 + EPS * np.random.normal(size=N)
    s1 = g1 + EPS * np.random.normal(size=N)

    # 2) Common source: S -> M and S -> G
    s2 = np.random.normal(size=N)
    m2 = s2 + EPS * np.random.normal(size=N)
    g2 = s2 + EPS * np.random.normal(size=N)

    # 3) Latent common cause: E -> M, E -> G, E -> S
    e = np.random.normal(size=N)
    m3 = e + EPS * np.random.normal(size=N)
    g3 = e + EPS * np.random.normal(size=N)
    s3 = e + EPS * np.random.normal(size=N)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(
        3,
        2,
        hspace=0.4,
        wspace=0.25,
        left=0.08,
        right=0.92,
        top=0.8,
        bottom=0.18,
    )

    ax_graph1 = fig.add_subplot(gs[0, 0])
    _plot_graph(ax_graph1, [("M", "G"), ("G", "S")], "Mass-driven scenario")

    m1_resid = _residuals(m1, g1)
    s1_resid = _residuals(s1, g1)

    ax_scatter1 = fig.add_subplot(gs[0, 1])
    ax_scatter1.scatter(m1_resid[:1000], s1_resid[:1000], alpha=0.3)
    ax_scatter1.set_xlabel(r"$M - \langle M|G \rangle$", fontsize=16)
    ax_scatter1.set_ylabel(r"$S - \langle S|G \rangle$", fontsize=16)
    ax_scatter1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    xlim = ax_scatter1.get_xlim()
    ylim = ax_scatter1.get_ylim()
    ax_scatter1.set_xlim(xlim[0] * 1.25, xlim[1] * 0.95)
    ax_scatter1.set_ylim(ylim[0] * 1.05, ylim[1] * 1.05)
    ax_scatter1.text(
        0.05,
        0.95,
        r"$S \perp M \mid G$",
        transform=ax_scatter1.transAxes,
        fontsize=16,
        va="top",
        ha="left",
    )

    ax_graph2 = fig.add_subplot(gs[1, 0])
    _plot_graph(ax_graph2, [("S", "M"), ("S", "G")], "Feedback-driven scenario")

    ax_graph3 = fig.add_subplot(gs[2, 0])
    _plot_graph(ax_graph3, [("E", "M"), ("E", "G"), ("E", "S")], "Common-cause scenario")

    m2_resid = _residuals(m2, g2)
    s2_resid = _residuals(s2, g2)

    ax_scatter_combined = fig.add_subplot(gs[1:3, 1])
    ax_scatter_combined.scatter(m2_resid[:1000], s2_resid[:1000], alpha=0.3)
    ax_scatter_combined.set_xlabel(r"$M - \langle M|G \rangle$", fontsize=16)
    ax_scatter_combined.set_ylabel(r"$S - \langle S|G \rangle$", fontsize=16)
    ax_scatter_combined.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_scatter_combined.text(
        0.05,
        0.95,
        r"$S \perp\!\!\!\/ \: M \mid G$",
        transform=ax_scatter_combined.transAxes,
        fontsize=16,
        va="top",
        ha="left",
    )

    fig.add_artist(
        plt.Line2D(
            [0.08, 0.92],
            [0.60, 0.60],
            transform=fig.transFigure,
            color="black",
            linewidth=1.5,
        )
    )

    out_dir = _resolve_paper_plots_dir()
    out_pdf = out_dir / "example_plots.pdf"
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved paper figure: {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
