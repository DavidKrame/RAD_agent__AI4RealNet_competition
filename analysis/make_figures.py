"""Draw the README figure from the stored records: completed scenarios with and without each component.

    python analysis/make_figures.py

Writes `docs/figures/mechanisms_light.png` and `docs/figures/mechanisms_dark.png`. Each row compares the agent
(`results/local_764/xdp_agent.json`) with the same agent with one component switched off
(`results/local_764/without_*.json`).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")

PAIRS = (  # label, run without the component, run with it; in the order of Table 1 of the paper
    ("AC-anchored dispatch program", "without_dispatch_program", "xdp_agent"),
    ("OptimCVXPY, all uses", "without_optimcvxpy", "xdp_agent"),
    ("Secured outages in the dispatch program", "without_secured_outages", "xdp_agent"),
    ("Local pool", "without_local_pool", "xdp_agent"),
    ("LODF line openings", "without_lodf_openings", "xdp_agent"),
    ("Opening of the withheld lines", "without_withholding", "xdp_agent"),
    ("Scheduled-outage look-ahead", "without_lookahead", "xdp_agent"),
    ("Margin preservation", "without_margin_preservation", "xdp_agent"),
    ("Extended-horizon safeguard", "without_safeguard", "xdp_agent"),
)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", without="#eb6834", with_="#2a78d6", connector="#c3c2b7"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", without="#d95926", with_="#3987e5", connector="#383835"),
}


def completed(name):
    with open(os.path.join(ROOT, "results", "local_764", name + ".json")) as fh:
        return sum(r["completed"] for r in json.load(fh)["rows"])


def draw(theme, data):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.0, 1.2 + 0.42 * len(data)), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ys = list(range(len(data)))[::-1]
    for y, (label, off, on) in zip(ys, data):
        ax.plot([off, on], [y, y], color=t["connector"], linewidth=2, zorder=1, solid_capstyle="round")
        ax.scatter([off], [y], s=90, color=t["without"], edgecolors=t["surface"], linewidths=2, zorder=3)
        ax.scatter([on], [y], s=90, color=t["with_"], edgecolors=t["surface"], linewidths=2, zorder=3)
        lo, hi = min(off, on), max(off, on)
        ax.text(lo - 2.2, y, f"{lo}", ha="right", va="center", fontsize=9, color=t["ink2"])
        ax.text(hi + 2.2, y, f"{hi}   (+{on - off})", ha="left", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks(ys)
    ax.set_yticklabels([d[0] for d in data], fontsize=10, color=t["ink"])
    lo = min(min(off, on) for _, off, on in data)
    hi = max(max(off, on) for _, off, on in data)
    ax.set_xlim(lo - 12, hi + 20)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlabel("Scenarios completed, out of 764 (local evaluation set)", fontsize=9, color=t["ink2"])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=t["without"], label="without the component"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=t["with_"], label="with the component")]
    leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
                    fontsize=9, handletextpad=0.3, columnspacing=1.6)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"mechanisms_{theme}.png")
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


def main():
    data = [(label, completed(off), completed(on)) for label, off, on in PAIRS]
    for theme in THEMES:
        print("wrote", draw(theme, data))


if __name__ == "__main__":
    main()
