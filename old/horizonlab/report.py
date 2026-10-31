from pathlib import Path
import numpy as np
import pandas as pd


def make_plots(df, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    valid = df[df.H_s.notna()]
    if valid.empty:
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    synthetic = df.kind.eq("synthetic_fixture").all()
    title = "SYNTHETIC FIXTURES — not NVIDIA dataset results" if synthetic else "PhysicalAI label replay — conditional clearance"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")
    for (exp, source), group in valid.groupby(["experiment", "reference_source"]):
        h = np.sort(group.H_s.to_numpy())
        label = exp.replace("_", " ")
        axes[0].step(h, np.arange(1, len(h) + 1) / len(h), where="post", label=label)
    axes[0].set(xlabel="Capped certified horizon H (seconds)", ylabel="Fraction of valid windows ≤ H",
                title="Horizon distribution (overlapping windows)", ylim=(0, 1.02))
    axes[0].grid(alpha=0.16)
    axes[0].legend(fontsize=8, loc="lower right")
    base = valid[valid.experiment == valid.experiment.iloc[0]]
    means = base.groupby("scenario").H_s.mean().sort_values()
    axes[1].barh([s.replace("_", " ") for s in means.index], means.values, color="#277b9d")
    axes[1].set(xlabel="Mean capped H (seconds)", title=f"First configuration: {base.experiment.iloc[0]}")
    axes[1].grid(axis="x", alpha=0.16)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output / "horizon_summary.png", dpi=160)
    plt.close(fig)
    # Separate scenario summaries retain zero horizons and invalid-window counts.
    df.groupby(["experiment", "reference_source", "scenario"], dropna=False).agg(
        windows=("H_s", "size"), valid_windows=("H_s", "count"), mean_capped_H_s=("H_s", "mean"),
        median_capped_H_s=("H_s", "median"),
    ).to_csv(output / "by_scenario.csv")
