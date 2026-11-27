#!/usr/bin/env python3
"""
make_tables.py - Regenerates the LaTeX tables of the paper from the CSV files in results/.

Usage (from the repository root):
    python scripts/make_tables.py [results_dir] [output_dir]          # defaults: results tables

Output files (output_dir):
    tab_models.tex          evaluated ASR configurations
    tab_main_results.tex    mean and standard deviation of each metric (raw output)
    tab_pervideo_rows.tex   rows of the per-video table (the paper uses a longtable environment)
    tab_subtitles.tex       compliance with the UNE 153010 subtitle limits by segmentation strategy
    tab_ablation.tex        effect of terminology normalization and initial prompt
    tab_punctuation.tex     punctuation restoration for Wav2Vec2
    tab_descriptive.tex     descriptive statistics with 95% bootstrap confidence intervals
    tab_friedman.tex        Friedman test, Kendall's W and sensitivity analysis
    tab_wilcoxon.tex        significant pairwise Wilcoxon comparisons (Holm correction)

The numbers are identical to those of the paper; captions and spacing may differ slightly from the
typeset version. The qualitative summary table of the paper is written by hand and is not generated here.
"""
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tables"
os.makedirs(OUT, exist_ok=True)

M = ["Tiny", "Base", "Small", "Wav2Vec2", "Medium", "Large-v1", "Large-v2", "Large-v3", "Large-v3-turbo"]
P = {"Tiny": 39, "Base": 74, "Small": 244, "Wav2Vec2": 317, "Medium": 769, "Large-v1": 1550, "Large-v2": 1550,
     "Large-v3": 1550, "Large-v3-turbo": 809}
REF_LABEL = "Referencia (Gold Standard)"
# (CSV column, label in the tables, decimals, higher is better)
MET = [("CER (↓)", "CER", 3, False), ("WER (↓)", "WER", 3, False), ("Punc-WER (↓)", "Punc-WER", 3, False),
       ("PC-WER (↓)", "PC-WER", 3, False), ("BERTScore (↑)", "BERTScore", 3, True),
       ("SeMaScore (↑)", "SeMaScore", 3, True), ("F. Huerta (Legibilidad)", "FH-Index", 1, True)]
ARROW = {False: r"($\downarrow$)", True: r"($\uparrow$)"}

d = pd.read_csv(f"{BASE}/results_per_video_raw.csv")
ref = d[d["Modelo Whisper"] == REF_LABEL].copy()
df = d[d["Modelo Whisper"] != REF_LABEL].copy()
for x in (df, ref):
    x["vid"] = x["Video / Contexto"].str.extract(r"(\d+)").astype(int)
df = df.sort_values(["vid", "Modelo Whisper"]).reset_index(drop=True)
srt = pd.read_csv(f"{BASE}/subtitle_metrics.csv")


def neg(s):
    """Typeset minus signs of negative numbers."""
    return re.sub(r"(?<=[ \[(&])-(?=\d)", "$-$", s)


def write(name, text):
    with open(f"{OUT}/{name}.tex", "w", encoding="utf-8") as f:
        f.write(neg(text) + "\n")
    print(f"  -> {OUT}/{name}.tex")


# ---------------------------------------------------------------- evaluated configurations
write("tab_models", r"""\begin{table}[!t]
\centering
\caption{Evaluated ASR configurations. Layers: encoder / decoder.}
\label{tab:models}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l>{\raggedright\arraybackslash}p{3.4cm}cr}
\hline
Configuration & Checkpoint & Layers & Params (M) \\
\hline
Tiny & Systran/\allowbreak faster-whisper-tiny & 4 / 4 & 39 \\
Base & Systran/\allowbreak faster-whisper-base & 6 / 6 & 74 \\
Small & Systran/\allowbreak faster-whisper-small & 12 / 12 & 244 \\
Medium & Systran/\allowbreak faster-whisper-medium & 24 / 24 & 769 \\
Large-v1 & Systran/\allowbreak faster-whisper-large-v1 & 32 / 32 & 1550 \\
Large-v2 & Systran/\allowbreak faster-whisper-large-v2 & 32 / 32 & 1550 \\
Large-v3 & Systran/\allowbreak faster-whisper-large-v3 & 32 / 32 & 1550 \\
Large-v3-turbo & mobiuslabsgmbh/\allowbreak faster-whisper-large-v3-turbo & 32 / 4 & 809 \\
Wav2Vec2 & jonatasgrosman/\allowbreak wav2vec2-large-xlsr-53-spanish + 5-gram KenLM & 24 / CTC & 317 \\
\hline
\end{tabular}
\end{table}""")

# ---------------------------------------------------------------- main results
g = df.groupby("Modelo Whisper")
best = {c: (g[c].mean().idxmax() if hi else g[c].mean().idxmin()) for c, _, _, hi in MET}
rows = []
for m in M:
    x = df[df["Modelo Whisper"] == m]
    t, r_ = x["Tiempo Inferencia (s)"], x["RTF"]
    cells = [m, str(P[m]), f"{t.mean():.1f} $\\pm$ {t.std():.1f}", f"{r_.mean():.2f} $\\pm$ {r_.std():.2f}"]
    for c, _, k, _ in MET:
        s = f"{x[c].mean():.{k}f} $\\pm$ {x[c].std():.{k}f}"
        cells.append("\\textbf{" + s + "}" if best[c] == m else s)
    rows.append(" & ".join(cells) + r" \\")
head = " & ".join(f"{lab} {ARROW[hi]}" for _, lab, _, hi in MET)
write("tab_main_results", r"""\begin{table*}[!t]
\centering
\caption{Results of the nine ASR configurations without post-processing (mean $\pm$ standard deviation, 15 videos). RTF: real-time factor. Best values in bold. The negative FH-Index of Wav2Vec2 is due to its lack of punctuation (Section~\ref{sec:r111}).}
\label{tab:main_results}
\footnotesize
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrccccccccc}
\hline
Model & Params (M) & Time (s) & RTF & """ + head + r""" \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}}
\end{table*}""")

# ---------------------------------------------------------------- per-video rows (longtable in the paper)
rows = [r"Video & Model & Params (M) & Time (s) & RTF & CER & WER & Punc-WER & BERTScore & SeMaScore & FH-Index \\",
        r"\hline"]
for v in range(1, 16):
    rf = ref[ref.vid == v].iloc[0]
    rows.append(f"Video {v} & Reference & -- & -- & -- & -- & -- & -- & -- & -- & {rf['F. Huerta (Legibilidad)']:.2f} \\\\")
    for m in M:
        x = df[(df.vid == v) & (df["Modelo Whisper"] == m)].iloc[0]
        rows.append(f"Video {v} & {m} & {P[m]} & {x['Tiempo Inferencia (s)']:.2f} & {x['RTF']:.3f} & "
                    f"{x['CER (↓)']:.4f} & {x['WER (↓)']:.4f} & {x['Punc-WER (↓)']:.4f} & "
                    f"{x['BERTScore (↑)']:.4f} & {x['SeMaScore (↑)']:.4f} & {x['F. Huerta (Legibilidad)']:.2f} \\\\")
    rows.append(r"\hline")
write("tab_pervideo_rows", "% Caption: Results of the nine ASR configurations in each video, without post-processing.\n"
      "% Reference: FH-Index of the reference transcription. RTF: real-time factor.\n" + "\n".join(rows))

# ---------------------------------------------------------------- subtitles
S = srt[srt.variant == "raw"]
rows = []
for m in M:
    c = [m, str(P[m])]
    for col in ["% cumple long+CPS", "% cumple todo", "% >max dur"]:
        for s in ["native", "rules", "une"]:
            c.append(f"{S[(S.model == m) & (S.seg == s)][col].mean():.1f}")
    rows.append(" & ".join(c) + r" \\")
write("tab_subtitles", r"""\begin{table*}[!t]
\centering
\caption{Percentage of subtitles meeting the UNE 153010:2012 limits with three segmentation strategies (mean of 15 videos). Length and CPS: at most two lines of 37 characters and 15 characters per second; All limits: also a display time between 1 and 6~s; $>$6~s: subtitles displayed for more than 6~s.}
\label{tab:subtitles}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrccccccccc}
\hline
 & & \multicolumn{3}{c}{Length and CPS (\%) ($\uparrow$)} & \multicolumn{3}{c}{All limits (\%) ($\uparrow$)} & \multicolumn{3}{c}{$>$6~s (\%) ($\downarrow$)} \\
Model & Params (M) & Native & Rules & UNE & Native & Rules & UNE & Native & Rules & UNE \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}""")

# ---------------------------------------------------------------- ablation
abl = pd.read_csv(f"{BASE}/ablation_postprocessing.csv")


def ab(m, metric, var):
    x = abl[(abl.Modelo == m) & (abl["Métrica"] == metric) & (abl.Variante == var)]
    return x.iloc[0]


rows = []
for m in ["Small", "Large-v3", "Large-v3-turbo"]:
    for var, lab in [("raw", "Raw"), ("norm", "+ Normalization"), ("prompt", "+ Initial prompt")]:
        c = [m if var == "raw" else "", lab]
        for metric in ["WER", "Punc-WER", "PC-WER", "Recall términos"]:
            k = 2 if metric == "Recall términos" else 3
            if var == "raw":
                c.append(f"{ab(m, metric, 'norm').media_base:.{k}f}")
            else:
                a = ab(m, metric, var)
                c.append(f"{a.media_variante:.{k}f} ({a.p_Holm:.2f})")
        rows.append(" & ".join(c) + r" \\")
    rows.append(r"\hline")
write("tab_ablation", r"""\begin{table}[!t]
\centering
\caption{Effect of the post-processing components (mean of 15 videos; $p_{Holm}$ versus raw output in parentheses). Term recall is computed on the 5 videos containing the domain terms.}
\label{tab:ablation}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccc}
\hline
Model & Variant & WER & Punc-WER & PC-WER & Term recall \\
\hline
""" + "\n".join(rows[:-1]) + r"""
\hline
\end{tabular}}
\end{table}""")

# ---------------------------------------------------------------- punctuation restoration
med = pd.read_csv(f"{BASE}/punctuation_restoration_means.csv").set_index("model")
eff = pd.read_csv(f"{BASE}/punctuation_restoration_effect.csv").set_index("Métrica")
keys = ["WER", "Punc-WER", "PC-WER", "FH", "FH-subt", "% cumple long+CPS"]
rows = []
for m in ["Wav2Vec2", "Wav2Vec2+PR", "Small", "Large-v3", "Large-v3-turbo"]:
    x = med.loc[m]
    rows.append(" & ".join([m.replace("+PR", " + PR"), f"{x.WER:.3f}", f"{x['Punc-WER']:.3f}", f"{x['PC-WER']:.3f}",
                            f"{x.FH:.1f}", f"{x['FH-subt']:.1f}", f"{x['% cumple long+CPS']:.1f}"]) + r" \\")
    if m == "Wav2Vec2+PR":
        rows.append(r"\multicolumn{1}{r}{$p_{Holm}$} & " + " & ".join(
            "--" if eff.loc[k, "p_Holm"] >= 0.999 else f"{eff.loc[k, 'p_Holm']:.4f}" for k in keys) + r" \\")
        rows.append(r"\hline")
write("tab_punctuation", r"""\begin{table}[!t]
\centering
\caption{Effect of punctuation restoration (PR) on Wav2Vec2, compared with three Whisper configurations (mean of 15 videos). FH-subt and Compl.: readability and percentage of compliant subtitles with UNE-based segmentation. $p_{Holm}$: Wav2Vec2 with versus without PR.}
\label{tab:r111}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\hline
Model & WER & Punc-WER & PC-WER & FH & FH-subt & Compl. (\%) \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}}
\end{table}""")

# ---------------------------------------------------------------- Friedman, Kendall's W and sensitivity
excl = sorted(df.groupby("vid")["WER (↓)"].min().loc[lambda s: s < 0.02].index)
rows = []
for c, lab, _, _ in MET:
    pv = df.pivot(index="vid", columns="Modelo Whisper", values=c)[M]
    st, p = friedmanchisquare(*[pv[m] for m in M]); W = st / (len(pv) * (len(M) - 1))
    pv2 = pv.drop(index=excl)
    st2, _ = friedmanchisquare(*[pv2[m] for m in M]); W2 = st2 / (len(pv2) * (len(M) - 1))
    rows.append(f"{lab} & {st:.2f} & {p:.1e} & {W:.3f} & {W2:.3f} \\\\".replace("e-", "e$-$"))
write("tab_friedman", r"""\begin{table}[!t]
\centering
\caption{Friedman test for each metric ($n=15$ videos, $k=9$) and Kendall's $W$. $W$ (sens.): after excluding the """
      + str(len(excl)) + r""" videos with WER $<0.02$.}
\label{tab:friedman}
\scriptsize
\begin{tabular}{lcccc}
\hline
Metric & $\chi^2_F$ & $p$ & $W$ & $W$ (sens.) \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}""")

# ---------------------------------------------------------------- descriptive statistics (95% bootstrap CI)
order = [MET[1], MET[0]] + MET[2:]            # WER first, as in the paper
rng = np.random.default_rng(42)               # one generator; draws in the order CER, WER, ... (as in the paper)
cells = {}
for c, lab, k, _ in MET:
    for m in M:
        x = df[df["Modelo Whisper"] == m].sort_values("vid")[c].values
        bm = rng.choice(x, (10000, len(x))).mean(1); lo, hi = np.quantile(bm, [.025, .975])
        q1, q3 = np.quantile(x, [.25, .75]); kk = 2 if lab == "FH-Index" else 4
        cells[(lab, m)] = (f"{x.mean():.{kk}f} & {x.std(ddof=1):.{kk}f} & {np.median(x):.{kk}f} & {q3 - q1:.{kk}f} & "
                           f"[{lo:.{kk}f}, {hi:.{kk}f}]")
blocks = []
for c, lab, _, hi in order:
    blocks.append("\n".join(f"{lab} {ARROW[hi]} & {m} & {cells[(lab, m)]} \\\\" for m in M))
write("tab_descriptive", r"""\begin{table*}[!t]
\centering
\caption{Descriptive statistics of each metric and ASR configuration (15 videos). CI: 95\% bootstrap confidence interval of the mean.}
\label{tab:desc}
\scriptsize
\renewcommand{\arraystretch}{1.0}
\begin{tabular}{llccccc}
\toprule
\textbf{Metric} & \textbf{Model} & \textbf{Mean} & \textbf{Std} & \textbf{Median} & \textbf{IQR} & \textbf{95\% CI} \\
\midrule
""" + "\n\\midrule\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\end{table*}""")

# ---------------------------------------------------------------- significant pairwise comparisons
w = pd.read_csv(f"{BASE}/stats/wilcoxon_holm.csv")


def block(c, lab, hi):
    x = w[(w.Metric == c) & (w.significant)].sort_values("p_Holm", kind="stable")
    lines = [f" & {r.A} vs. {r.B} & {r.p_Holm:.4f} & {r.r:.2f} \\\\" for r in x.itertuples()]
    lines[0] = f"\\multirow{{{len(lines)}}}{{*}}{{{lab} {ARROW[hi]}}}\n" + lines[0]
    return "\n".join(lines)


def half(mets):
    return (r"""\begin{tabular}{@{}l l c c@{}}
\toprule
\textbf{Metric} & \textbf{Pairwise comparison} & \textbf{Holm $p$} & \textbf{$r$} \\
\midrule
""" + "\n\\midrule\n".join(block(c, lab, hi) for c, lab, _, hi in mets) + "\n\\bottomrule\n\\end{tabular}")


left, right = [MET[1], MET[0], MET[2]], [MET[3], MET[4], MET[5], MET[6]]
write("tab_wilcoxon", r"""\begin{table*}[!t]
\centering
\caption{Pairwise Wilcoxon comparisons significant after Holm correction ($p_{Holm}<0.05$), with effect size $r$.}
\label{tab:wilcoxon}
\scriptsize
\renewcommand{\arraystretch}{1.0}
\begin{minipage}[t]{0.49\textwidth}
\centering
""" + half(left) + r"""
\end{minipage}\hfill
\begin{minipage}[t]{0.49\textwidth}
\centering
""" + half(right) + r"""
\end{minipage}
\end{table*}""")
print("Significant pairs:", int(w.significant.sum()), "of", len(w), "| sensitivity excludes videos", excl)
