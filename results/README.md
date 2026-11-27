# Results

Frozen results reported in the paper. Column names are kept as produced by the evaluation notebook
(some of them in Spanish); this file explains them. Model names: Tiny, Base, Small, Medium, Large-v1,
Large-v2, Large-v3, Large-v3-turbo (Whisper) and Wav2Vec2. "Wav2Vec2+PR" is Wav2Vec2 with punctuation
restoration.

## results_per_video_raw.csv

One row per video (1-15) and configuration, on the raw ASR output (no post-processing), plus one row per
video for the reference transcription (`Referencia (Gold Standard)`, used only for its FH-Index).

| Column | Meaning |
|---|---|
| Video / Contexto | Video identifier (Video 1 ... Video 15) |
| Modelo Whisper | ASR configuration |
| Parámetros | Number of parameters |
| Tiempo Inferencia (s) | Processing time of the complete file (s), Apple M4 CPU |
| CER, WER (↓) | Character and word error rate (lowercase, without punctuation) |
| Punc-WER (↓) | WER with punctuation marks as separate tokens (lowercase) |
| PC-WER (↓) | Punc-WER that also distinguishes upper and lower case |
| BERTScore (↑) | BERTScore F1 (XLM-RoBERTa large, layer 17, no rescaling), on aligned fragments of 10-60 words |
| SeMaScore (↑) | SeMaScore (official algorithm, same encoder), on the same fragments |
| SBERT-cos (↑) | Cosine similarity with paraphrase-multilingual-MiniLM-L12-v2 (supplementary) |
| F. Huerta (Legibilidad) | Fernández-Huerta readability index of the complete text (textstat 0.7.13) |
| RTF | Real-time factor = processing time / audio duration |
| Recall términos | Proportion of domain-term occurrences recognized (only CEACOG appears in the references; 5 videos) |
| Ratio palabras | Number of words of the transcription / number of words of the reference |
| Idioma | Language used for decoding (fixed to Spanish) |
| Ventanas SeMa fallidas | SeMaScore fragments that could not be aligned (0 in all cases) |
| Archivo | ASR output file |

## subtitle_metrics.csv

One row per video, model, output variant (`raw`, `norm`, `prompt`, `prompt+norm`, `punct`) and segmentation
(`native`: segments of the model; `rules`: rule-based, target 5 s and maximum 7 s; `une`: UNE 153010-based).

| Column | Meaning |
|---|---|
| Subtítulos | Number of subtitles |
| Car/subt | Mean characters per subtitle |
| Dur media (s) | Mean display time (s) |
| CPS mediana, CPS p90 | Median and 90th percentile of characters per second |
| % CPL≤max (tal cual) | % of subtitles already written in at most 2 lines of 37 characters |
| % cabe en líneas | % of subtitles whose text fits in at most 2 lines of 37 characters |
| % CPS≤max | % of subtitles with at most 15 CPS |
| % duración en rango | % of subtitles displayed between 1 and 6 s |
| % <min dur, % >max dur | % of subtitles shorter than 1 s / longer than 6 s |
| % cumple long+CPS | % meeting the length and CPS limits ("Length and CPS") |
| % cumple todo | % also meeting the display-time limits ("All limits") |
| % solapados | % of subtitles overlapping the previous one |
| FH-subt | Mean Fernández-Huerta index of the subtitles |

## Ablation

* `ablation_postprocessing.csv`: terminology normalization (`norm`), initial prompt (`prompt`, `prompt+norm`)
  and, for Wav2Vec2, punctuation restoration (`punct`), compared with the raw output (`Base`=raw).
* `ablation_segmentation.csv`: `rules` and `une` segmentations compared with `native`.

Columns: model (`Modelo`), metric (`Métrica`), baseline (`Base`) and variant (`Variante`), their means
(`media_base`, `media_variante`), mean difference (`delta`), Wilcoxon p-value (`p_raw`), number of videos (`n`)
and Holm-corrected p-value within each metric (`p_Holm`).

## Punctuation restoration (Wav2Vec2)

* `punctuation_restoration_means.csv`: means of WER, Punc-WER, PC-WER, FH, FH-subt and compliance
  (UNE-based segmentation) for each configuration, including Wav2Vec2+PR.
* `punctuation_restoration_effect.csv`: Wav2Vec2 without versus with restoration (Wilcoxon, Holm).
* `punctuation_restoration_vs_whisper.csv`: Wav2Vec2+PR versus each Whisper model (`Comparado`), with the
  Friedman p-value, Wilcoxon p-values, effect size r and significance after Holm correction.

## stats/

* `friedman.csv`: Friedman test for each metric (n = 15 videos, k = 9 configurations) and Kendall's W.
* `wilcoxon_holm.csv`: 36 pairwise Wilcoxon signed-rank tests (Pratt) per metric, Holm correction within each
  metric, effect size r = |Z| / sqrt(N).
* `descriptive.csv`: mean, standard deviation, median, interquartile range and 95% bootstrap confidence
  interval of the mean (10,000 resamples, seed 42).
* `sensitivity.csv`: Friedman p-value, Kendall's W and mean ranks with all videos (`todos`) and after
  excluding the 5 videos where any model obtained WER < 0.02 (videos 2, 11, 12, 13 and 14).

## logs/

* `run_log.csv`: one row per ASR run (model, decoding, video, processing time, audio duration, RTF,
  segments, words and segments decoded again by temperature fallback).
* `punctuation_log.csv`: punctuation restoration of each Wav2Vec2 transcript (words, words without a 1:1
  mapping, time).
