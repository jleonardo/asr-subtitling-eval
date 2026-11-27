#!/usr/bin/env python3
"""
punctuation_restoration_wav2vec.py - Punctuation and true-casing restoration for the Wav2Vec 2.0 output.

Wav2Vec 2.0 produces neither punctuation nor capitalization. This script restores them with an external
model, so the complete system Wav2Vec2 + restoration (Wav2Vec2+PR) can be compared with Whisper using its
native punctuation. By default only wav2vec2 is processed (the Whisper outputs are not modified).

Default model: 1-800-BAD-CODE/punctuation_fullstop_truecase_romance (package `punctuators`, ONNX, CPU):
punctuation (. , ? ¿), capitalization and sentence boundaries for Spanish.
The original word timestamps are kept, so the subtitles can be re-segmented.

Usage (from the repository root):
    pip install punctuators
    python scripts/punctuation_restoration_wav2vec.py --out data/asr_outputs --dry-run   # should list 15 jobs
    python scripts/punctuation_restoration_wav2vec.py --out data/asr_outputs

New outputs in <out>/VIDEO-N/ (variant 'punct', read by the evaluation notebook):
    wav2vec2__punct.txt   wav2vec2__punct__native.srt   wav2vec2__punct__une.srt
and <out>/registro_puntuacion.csv (restoration time and quality of the word mapping).
Existing raw/norm/prompt files are never modified or deleted.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import time
from datetime import datetime
from importlib.metadata import version as pkg_version, PackageNotFoundError
from pathlib import Path

DEFAULT_MODEL = "1-800-BAD-CODE/punctuation_fullstop_truecase_romance"
MODELS = ["wav2vec2"]          # only Wav2Vec2 by default (use --models to add others)
CHUNK_WORDS = 150          # words per block sent to the model


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def plain(token: str) -> str:
    """Lowercase form without punctuation (what the model expects as input)."""
    return re.sub(r"[^\w]", "", token.lower())


def read_header(path: Path) -> dict:
    head = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            break
        k, _, v = line.partition(":")
        head[k.strip()] = v.strip()
    return head


def flat_words(segments: list) -> list:
    """Words with timestamps, in order, with their 'plain' form. Tokens without letters or digits are dropped."""
    out = []
    for si, s in enumerate(segments):
        ws = s.get("words") or []
        if not ws and s.get("text", "").strip():       # segment without words: uniform split
            toks = s["text"].split(); step = (s["end"] - s["start"]) / max(len(toks), 1)
            ws = [{"word": t, "start": s["start"] + i * step, "end": s["start"] + (i + 1) * step}
                  for i, t in enumerate(toks)]
        for w in ws:
            p = plain(w["word"])
            if p:
                out.append({"seg": si, "start": w["start"], "end": w["end"], "plain": p})
    return out


def restore(model, plains: list) -> tuple[list, int]:
    """Return a list of punctuated tokens with the SAME length as `plains`, and the number of words without a 1:1 mapping."""
    restored = []
    for i in range(0, len(plains), CHUNK_WORDS):
        chunk = plains[i:i + CHUNK_WORDS]
        sentences = model.infer(texts=[" ".join(chunk)], apply_sbd=True)[0]
        toks = " ".join(sentences).split()
        if [plain(t) for t in toks] == chunk:
            restored += toks
            continue
        # Align by plain form if the model changed some token (e.g., acronyms)
        out = list(chunk)
        sm = difflib.SequenceMatcher(a=chunk, b=[plain(t) for t in toks], autojunk=False)
        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            if tag == "equal":
                out[a0:a1] = toks[b0:b1]
            elif tag == "replace" and (a1 - a0) == (b1 - b0):
                out[a0:a1] = toks[b0:b1]
        restored += out
    unmapped = sum(1 for r, p in zip(restored, plains) if plain(r) != p)
    return restored, unmapped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="folder with the ASR outputs (VIDEO-N subfolders)")
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rerun_controlled import write_srt, write_txt, seg_native, seg_une

    jobs = []
    for vdir in sorted(args.out.glob("VIDEO-*"), key=lambda p: int(p.name.split("-")[1])):
        for m in args.models:
            js, txt = vdir / f"{m}__raw.words.json", vdir / f"{m}__raw.txt"
            if js.exists() and txt.exists() and (args.force or not (vdir / f"{m}__punct.txt").exists()):
                jobs.append((vdir, m, js, txt))
    log(f"Punctuation model: {args.model_id} | pending jobs: {len(jobs)}")
    if args.dry_run or not jobs:
        return

    from punctuators.models import PunctCapSegModelONNX
    t0 = time.perf_counter()
    model = PunctCapSegModelONNX.from_pretrained(args.model_id)
    log(f"Model loaded in {time.perf_counter() - t0:.1f} s")
    model.infer(texts=["esto es una prueba de calentamiento"], apply_sbd=True)

    reg_path = args.out / "registro_puntuacion.csv"
    new = not reg_path.exists()
    fh = reg_path.open("a", newline="", encoding="utf-8"); wr = csv.writer(fh)
    if new:
        wr.writerow(["fecha", "modelo", "video", "palabras", "sin_mapeo_1a1", "tiempo_s", "duracion_audio_s", "rtf_puntuacion"])

    for vdir, m, js, txt in jobs:
        segments = json.loads(js.read_text(encoding="utf-8"))
        words = flat_words(segments)
        t = time.perf_counter()
        restored, unmapped = restore(model, [w["plain"] for w in words])
        elapsed = time.perf_counter() - t

        # Rebuild the segments with the punctuated words (same timestamps)
        new_segs = [{"start": s["start"], "end": s["end"], "text": "", "words": []} for s in segments]
        for w, tok in zip(words, restored):
            new_segs[w["seg"]]["words"].append({"word": " " + tok, "start": w["start"], "end": w["end"]})
        for s in new_segs:
            s["text"] = "".join(x["word"] for x in s["words"]).strip()
        new_segs = [s for s in new_segs if s["text"]]

        header = read_header(txt)
        dur = float(re.search(r"([\d.]+)", header.get("Duración audio", "nan")).group(1)) if header.get("Duración audio") else float("nan")
        header.update({"Variante": "punct", "Normalización": "no",
                       "Puntuación": f"restaurada con {args.model_id} (punctuators {pkg_v('punctuators')})",
                       "Tiempo restauración puntuación": f"{elapsed:.2f} segundos",
                       "Palabras sin mapeo 1a1": unmapped,
                       "Palabras transcritas": len(restored)})
        # Text: one sentence per line (sentence boundaries predicted by the model)
        body = re.sub(r"([.?!])\s+", r"\1\n", " ".join(restored)).splitlines()
        write_txt(vdir / f"{m}__punct.txt", header, body)
        write_srt(vdir / f"{m}__punct__native.srt", seg_native(new_segs))
        write_srt(vdir / f"{m}__punct__une.srt", seg_une(new_segs))
        wr.writerow([datetime.now().isoformat(timespec="seconds"), m, vdir.name, len(restored), unmapped,
                     f"{elapsed:.3f}", f"{dur:.2f}", f"{elapsed / dur:.5f}" if dur == dur else ""]); fh.flush()
        log(f"{m:15s} {vdir.name:9s} words={len(restored):4d} unmapped={unmapped:2d} t={elapsed:5.2f} s")
    fh.close()
    log("Finished.")


def pkg_v(p):
    try:
        return pkg_version(p)
    except PackageNotFoundError:
        return "?"


if __name__ == "__main__":
    main()
