#!/usr/bin/env python3
"""
make_figures.py - Regenerates the figures of the paper from the CSV files in results/.

    Fig. 3  Distribution of WER, CER, BERTScore and SeMaScore (boxplots)
    Fig. 4  Mean WER versus mean real-time factor (RTF), with 95% bootstrap confidence intervals
    Fig. 5  Percentage of subtitles meeting the length and reading-speed limits by segmentation strategy

It only reads the frozen CSV files; it does not recompute any metric.
Usage (from the repository root):
    python scripts/make_figures.py [results_dir] [output_dir]        # defaults: results figures
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["Tiny", "Base", "Small", "Wav2Vec2", "Medium", "Large-v1", "Large-v2", "Large-v3", "Large-v3-turbo"]
REF_LABEL = "Referencia (Gold Standard)"

df = pd.read_csv(f"{BASE}/results_per_video_raw.csv")
df = df[df["Modelo Whisper"] != REF_LABEL].copy()
assert sorted(df["Modelo Whisper"].unique()) == sorted(MODELS), "Models missing in the CSV"
assert df.groupby("Modelo Whisper").size().eq(15).all(), "There are not 15 videos per model"
df["vid"] = df["Video / Contexto"].str.extract(r"(\d+)").astype(int)
srt = pd.read_csv(f"{BASE}/subtitle_metrics.csv")

RNG = np.random.default_rng(42)


def boot_ci(x, n=10_000, alpha=0.05):
    """Mean and 95% percentile bootstrap confidence interval (fixed seed)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    means = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)


def save(fig, name, dpi=600):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_DIR}/{name}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  -> {OUT_DIR}/{name}.png / .pdf")


# =============================================================================
# Fig. 3 - Boxplots of lexical and semantic metrics
# =============================================================================
print("Fig. 3 - boxplots")
d3 = df.copy()
medianprops = dict(color="black", linewidth=1.5)
meanprops = dict(marker="D", markeredgecolor="black", markerfacecolor="firebrick", markersize=6)
panels = [("WER (↓)", "Blues", "WER Score Distribution – Models (↓ better)", "Word Error Rate (WER)"),
          ("BERTScore (↑)", "Greens", "BERTScore Score Distribution – Models (↑ better)", "BERTScore"),
          ("CER (↓)", "ch:start=.2,rot=-.3", "CER Score Distribution – Models (↓ better)", "Character Error Rate (CER)"),
          ("SeMaScore (↑)", "crest", "SeMaScore Score Distribution – Models (↑ better)", "SeMaScore")]
fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=False)
for ax, (col, pal, title, ylab) in zip(axes.flatten(), panels):
    sns.boxplot(x="Modelo Whisper", y=col, data=d3, ax=ax, palette=pal, hue="Modelo Whisper", legend=False,
                order=MODELS, hue_order=MODELS, dodge=False,
                medianprops=medianprops, meanprops=meanprops, showmeans=True)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=14, fontweight="bold")
    ax.set_xlabel("Models", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    ax.xaxis.grid(True, linestyle="--", alpha=0.2)
plt.suptitle("Multidimensional Comparison: Lexical Error vs. Semantic Preservation", fontsize=18, y=1.02,
             fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.legend(handles=[Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
                    Line2D([0], [0], marker="D", color="firebrick", linestyle="None", markersize=8,
                           markeredgecolor="black", label="Mean")],
           loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=10, frameon=True)
save(fig, "fig3_boxplots")

# =============================================================================
# Common style of Figs. 4 and 5
# =============================================================================
COLOR = {"Tiny": "#2a78d6", "Base": "#eb6834", "Small": "#1baf7a", "Wav2Vec2": "#eda100",
         "Medium": "#e87ba4", "Large-v1": "#6f6e69", "Large-v2": "#008300",
         "Large-v3": "#6250d6", "Large-v3-turbo": "#e34948"}
MARKER = {"Tiny": "o", "Base": "s", "Small": "^", "Wav2Vec2": "*", "Medium": "D",
          "Large-v1": "h", "Large-v2": "v", "Large-v3": "P", "Large-v3-turbo": "X"}
SEG_COLOR = {"native": "#2a78d6", "rules": "#eb6834", "une": "#1baf7a"}
SEG_LABEL = {"native": "Native (model output)", "rules": "Rule-based (max. 7 s)", "une": "UNE 153010-based"}
INK, INK2, GRID = "#1f1f1f", "#5f5f5f", "#e6e6e6"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 150,
                     "savefig.dpi": 600, "axes.spines.top": False, "axes.spines.right": False})

# =============================================================================
# Fig. 4 - Quality versus speed
# =============================================================================
print("Fig. 4 - WER vs. RTF (mean and 95% bootstrap CI; x = maximum per-video RTF)")
rows = []
for m in MODELS:
    g = df[df["Modelo Whisper"] == m]
    w, wlo, whi = boot_ci(g["WER (↓)"])
    r, rlo, rhi = boot_ci(g["RTF"])
    rows.append(dict(model=m, WER=w, WER_lo=wlo, WER_hi=whi, RTF=r, RTF_lo=rlo, RTF_hi=rhi,
                     RTF_max=g["RTF"].max(), n_rtf_gt1=int((g["RTF"] > 1).sum())))
f4 = pd.DataFrame(rows)
print(f4.round(3).to_string(index=False))
f4.to_csv(f"{OUT_DIR}/fig4_quality_vs_speed_data.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.axvspan(1, 3, color="#f4f4f4", zorder=0)
ax.axvline(1, color=INK2, lw=1, ls="--", zorder=1)
ax.text(1.04, 0.335, "slower than\nreal time (RTF > 1)", fontsize=8, color=INK2, va="top")
for _, r in f4.iterrows():
    c = COLOR[r.model]
    ax.errorbar(r.RTF, r.WER, xerr=[[r.RTF - r.RTF_lo], [r.RTF_hi - r.RTF]],
                yerr=[[r.WER - r.WER_lo], [r.WER_hi - r.WER]],
                fmt="none", ecolor=c, elinewidth=1.2, capsize=2.5, alpha=0.8, zorder=2)
    ax.scatter(r.RTF, r.WER, s=90 if MARKER[r.model] != "*" else 160, marker=MARKER[r.model],
               color=c, edgecolor="white", linewidth=1.2, zorder=3)
    ax.scatter(r.RTF_max, r.WER, s=28, marker="x", color=c, linewidth=1.1, zorder=3)
OFF = {"Tiny": (8, 6), "Base": (8, 6), "Small": (-8, -14), "Wav2Vec2": (8, 6),
       "Medium": (-10, 8), "Large-v1": (-6, 22), "Large-v2": (4, 24),
       "Large-v3": (6, -18), "Large-v3-turbo": (-10, -16)}
for _, r in f4.iterrows():
    dx, dy = OFF[r.model]
    ax.annotate(r.model, (r.RTF, r.WER), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, color=INK, ha="left" if dx >= 0 else "right",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5) if abs(dy) > 15 else None)
ax.set_xscale("log")
ax.set_xlim(0.02, 2.6)
ax.set_ylim(0, 0.40)
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
ax.set_xlabel("Real-time factor, RTF (log scale; lower is faster)")
ax.set_ylabel("WER (lower is better)")
ax.grid(True, color=GRID, lw=0.6, zorder=0)
ax.legend(handles=[Line2D([0], [0], marker="o", ls="none", color=INK2, label="Mean (95% bootstrap CI)"),
                   Line2D([0], [0], marker="x", ls="none", color=INK2, label="Maximum per-video RTF")],
          loc="upper center", frameon=False)
fig.tight_layout()
save(fig, "fig4_quality_vs_speed")

# =============================================================================
# Fig. 5 - Subtitle compliance (at most 2 lines of 37 characters and at most 15 CPS)
# =============================================================================
print("Fig. 5 - % of subtitles meeting length and CPS limits (mean and 95% bootstrap CI)")
S = srt[srt["variant"] == "raw"]
rows = []
for m in MODELS:
    for s in ["native", "rules", "une"]:
        x = S[(S["model"] == m) & (S["seg"] == s)]["% cumple long+CPS"]
        mu, lo, hi = boot_ci(x)
        rows.append(dict(model=m, seg=s, mean=mu, lo=lo, hi=hi, n=len(x)))
f5 = pd.DataFrame(rows)
print(f5.pivot(index="model", columns="seg", values="mean").reindex(MODELS)[["native", "rules", "une"]].round(1))
f5.to_csv(f"{OUT_DIR}/fig5_subtitle_compliance_data.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 3.9))
w, x0 = 0.26, np.arange(len(MODELS))
for k, s in enumerate(["native", "rules", "une"]):
    d = f5[f5["seg"] == s].set_index("model").reindex(MODELS)
    ax.bar(x0 + (k - 1) * w, d["mean"], width=w - 0.03, color=SEG_COLOR[s], label=SEG_LABEL[s], zorder=2)
    ax.errorbar(x0 + (k - 1) * w, d["mean"], yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
ax.set_xticks(x0)
ax.set_xticklabels(MODELS)
ax.set_ylim(0, 100)
ax.set_ylabel("Subtitles meeting constraints (%)")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, title="Segmentation strategy")
fig.tight_layout()
save(fig, "fig5_subtitle_compliance")
print("Done.")
