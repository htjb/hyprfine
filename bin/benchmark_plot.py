"""Plot benchmark results from one or more JSON files produced by benchmark_time.py.

Usage:
    python bin/benchmark_plot.py FILE [FILE ...] [--output benchmark.png]

Each FILE is a JSON saved by benchmark_time.py. The signal panel is taken from
the first file; the throughput panel shows all hardware entries overlaid.
"""

import argparse
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("files", nargs="+", help="JSON result files")
parser.add_argument("--output", default="bin/benchmark.png")
args = parser.parse_args()

datasets = []
for path in args.files:
    with open(path) as fh:
        datasets.append(json.load(fh))

COLOURS = ["steelblue", "darkorange", "crimson", "purple", "olive", "teal"]

fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3)
ax_sig = fig.add_subplot(gs[0, :2])
ax_batch = fig.add_subplot(gs[0, 2])

# ---------------------------------------------------------------------------
# Signal panel — use first dataset
# ---------------------------------------------------------------------------
d0 = datasets[0]
z_grid = 1420.4 / np.array(d0["f_grid"]) - 1
ax_sig.plot(z_grid, d0["signal"], color="steelblue", lw=1.5, label="hyprfine")
ax_sig.plot(
    d0["z21_zintegral"], d0["z21_signal"],
    color="seagreen", lw=1.5, ls="--", label="zeus21",
)
ax_sig.set_xlabel("Redshift $z$")
ax_sig.set_ylabel(r"$T_{21}$ [mK]")
ax_sig.set_title("21-cm signal")
ax_sig.set_xscale("log")
ax_sig.legend(fontsize=8)

# ---------------------------------------------------------------------------
# Throughput panel — one entry per dataset
# ---------------------------------------------------------------------------
for i, d in enumerate(datasets):
    colour = COLOURS[i % len(COLOURS)]
    batch_sizes = d["batch_sizes"]

    ax_batch.plot(
        batch_sizes, d["cpu_batch_times"],
        color=colour, marker="o", lw=1.5, label=f"hyprfine ({d['cpu_name']})",
    )
    if d["gpu_batch_times"] is not None:
        ax_batch.plot(
            batch_sizes, d["gpu_batch_times"],
            color=colour, marker="s", lw=1.5, ls="--",
            label=f"hyprfine ({d['gpu_name']})",
        )
    ax_batch.axhline(
        d["z21_time"], color=colour, ls=":", lw=1.2,
        label=f"zeus21 ({d['cpu_name']})",
    )

ax_batch.set_xlabel("Batch size")
ax_batch.set_ylabel("Time per signal [s]")
ax_batch.set_title("Throughput scaling")
ax_batch.set_xscale("log", base=2)
ax_batch.set_yscale("log")
bs0 = datasets[0]["batch_sizes"]
ax_batch.set_xticks(bs0)
ax_batch.set_xticklabels([str(b) for b in bs0])
ax_batch.legend(fontsize=7)

plt.savefig(args.output, dpi=300)
print(f"Saved {args.output}")
