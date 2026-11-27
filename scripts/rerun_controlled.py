#!/usr/bin/env python3
"""
rerun_controlled.py - Controlled, reproducible ASR runs used in the paper.

Usage (from the repository root, with the virtual environment activated):

    python scripts/rerun_controlled.py --videos data/videos --out data/asr_outputs --dry-run
    python scripts/rerun_controlled.py --videos data/videos --out data/asr_outputs --check-determinism
    python scripts/rerun_controlled.py --videos data/videos --out data/asr_outputs

--videos must contain one file per video whose name includes VIDEO-N (mp4, wav, ...).

Controlled configuration (identical for all Whisper models):
  * faster-whisper, CPU, compute_type=int8, beam_size=5
  * language="es" (no automatic language detection)
  * temperature: default Whisper fallback (0.0 -> 1.0) with a fixed CTranslate2 seed set before each
    transcription, so runs are reproducible and the anti-loop mechanism is kept
  * word_timestamps=True (needed for the rule-based and UNE-based segmentations)
  * audio decoded to 16 kHz mono BEFORE timing starts (same for Wav2Vec2)
  * each model is warmed up with 10 s of audio (not timed)

Outputs in <out>/VIDEO-N/ (format read by notebooks/evaluation_metrics.ipynb):
  <model>__<variant>.txt                    "Key: value" header + blank line + transcript
  <model>__<variant>__<segmentation>.srt    segmentations: native, rules, une
  <model>__<decoding>.words.json            words with timestamps (to re-segment without re-running ASR)
Variants: raw (raw output), norm (raw + dictionary normalization), prompt (initial_prompt), prompt+norm.
Also: <out>/entorno.json (hardware and package versions) and <out>/registro_ejecucion.csv (run log).

Note: the header keys of the .txt files and the log column names are kept in Spanish, as in the
original runs, because the evaluation notebook parses them (e.g., "Tiempo transcurrido", "Duración audio").
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version as pkg_version, PackageNotFoundError
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"   # custom_normalizations, whisper_prompts, wav2vec_transcriber

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]
PROMPT_MODELS = ["small", "large-v3", "large-v3-turbo"]      # models with a 'prompt' variant (ablation)
COMPUTE_TYPE = "int8"
DEVICE = "cpu"
BEAM_SIZE = 5
LANGUAGE = "es"
TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)                # faster-whisper default
SEED = 42
WARMUP_SECONDS = 10.0
SR = 16000

# 'rules' segmentation = split_segment_for_subtitles of the transcription application (default values)
RULES_MAX_DURATION, RULES_TARGET_DURATION = 7.0, 5.0
# 'une' segmentation: limits derived from UNE 153010:2012
UNE_MAX_CPL, UNE_MAX_LINES, UNE_MAX_CPS = 37, 2, 15.0
UNE_MIN_DUR, UNE_MAX_DUR, UNE_MIN_GAP, UNE_PAUSE = 1.0, 6.0, 0.08, 0.7

CHECKPOINTS = {  # faster-whisper 1.2.1 (utils._MODELS)
    "tiny": "Systran/faster-whisper-tiny", "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small", "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2", "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def video_id(name: str):
    m = re.search(r"VIDEO-(\d+)", name, re.I)
    return int(m.group(1)) if m else None


def pv(pkg: str) -> str:
    try:
        return pkg_version(pkg)
    except PackageNotFoundError:
        return "not installed"


def sysctl(key: str) -> str:
    """macOS hardware information (returns an empty string on other systems)."""
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def environment_info(args) -> dict:
    mem = sysctl("hw.memsize")
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "ram_GB": round(int(mem) / 2**30, 1) if mem.isdigit() else None,
        "nucleos_fisicos": sysctl("hw.physicalcpu"),
        "nucleos_rendimiento": sysctl("hw.perflevel0.physicalcpu"),
        "nucleos_eficiencia": sysctl("hw.perflevel1.physicalcpu"),
        "macOS": platform.mac_ver()[0], "python": platform.python_version(),
        "paquetes": {p: pv(p) for p in ["faster-whisper", "ctranslate2", "torch", "transformers",
                                        "pyctcdecode", "kenlm", "numpy"]},
        "config": {"device": DEVICE, "compute_type": COMPUTE_TYPE, "beam_size": BEAM_SIZE,
                   "language": LANGUAGE, "temperatures": TEMPERATURES, "seed": SEED,
                   "word_timestamps": True, "vad_filter": False,
                   "condition_on_previous_text": True, "cpu_threads": args.cpu_threads,
                   "warmup_s": WARMUP_SECONDS, "checkpoints": CHECKPOINTS,
                   "rules": {"max_duration": RULES_MAX_DURATION, "target_duration": RULES_TARGET_DURATION},
                   "une": {"max_cpl": UNE_MAX_CPL, "max_lines": UNE_MAX_LINES, "max_cps": UNE_MAX_CPS,
                           "min_dur": UNE_MIN_DUR, "max_dur": UNE_MAX_DUR, "pause_split": UNE_PAUSE}},
        "argumentos": vars(args) | {"videos": str(args.videos), "out": str(args.out)},
    }


def set_seed(ct2) -> None:
    if hasattr(ct2, "set_random_seed"):
        ct2.set_random_seed(SEED)
    else:
        log("WARNING: this ctranslate2 version has no set_random_seed; temperature fallback will not be reproducible")


def to_wav16k(src: Path, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1",
                    "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)], check=True)
    return dst


def load_wav(path: Path):
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return a


def srt_time(s: float) -> str:
    ms = int(round(max(s, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def write_srt(path: Path, subs: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(subs, 1):
            f.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n\n")


def write_txt(path: Path, header: dict, lines: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for k, v in header.items():
            f.write(f"{k}: {str(v).replace(chr(10), ' ')}\n")
        f.write("\n")
        f.write("\n".join(l for l in lines if l.strip()))
        f.write("\n")


# ---------------------------------------------------------------------------
# Segmentations
# ---------------------------------------------------------------------------
def seg_native(segments: list) -> list:
    """Segments as produced by the model."""
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in segments if s["text"].strip()]


def _join_words(ws: list) -> str:
    """faster-whisper includes the leading space in each word; Wav2Vec2 does not."""
    return "".join(w if w.startswith(" ") else " " + w for w in ws).strip()


def seg_rules(segments: list) -> list:
    """Replica of split_segment_for_subtitles of the transcription application
    (max_duration=7.0, target_duration=5.0), applied to each segment."""
    out = []
    for s in segments:
        words = s.get("words")
        if not words or (s["end"] - s["start"]) <= RULES_MAX_DURATION:
            if s["text"].strip():
                out.append({"start": s["start"], "end": s["end"], "text": s["text"].strip()})
            continue
        cur, cs, ce, sub = [], None, None, []
        for w in words:
            if cs is None:
                cs = w["start"]
            cur.append(w["word"]); ce = w["end"]
            if ce - cs >= RULES_TARGET_DURATION:
                t = _join_words(cur)
                if t:
                    sub.append({"start": cs, "end": ce, "text": t})
                cur, cs, ce = [], None, None
        if cur:
            t = _join_words(cur)
            if t:
                sub.append({"start": cs if cs is not None else s["start"],
                            "end": ce if ce is not None else s["end"], "text": t})
        out += sub or [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}]
    return out


def _wrap(text: str, width: int) -> list:
    lines, cur = [], ""
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def _balanced_lines(text: str, width: int, max_lines: int) -> str:
    """Split into <= max_lines lines of <= width characters; with two lines, choose the most balanced cut."""
    if len(text) <= width:
        return text
    words = text.split()
    if max_lines >= 2:
        best = None
        for i in range(1, len(words)):
            a, b = " ".join(words[:i]), " ".join(words[i:])
            if len(a) <= width and len(b) <= width:
                score = abs(len(a) - len(b)) - (3 if a.endswith((",", ".", ";", ":", "?", "!")) else 0)
                if best is None or score < best[0]:
                    best = (score, a, b)
        if best:
            return best[1] + "\n" + best[2]
    return "\n".join(_wrap(text, width))


def seg_une(segments: list) -> list:
    """UNE 153010-based segmentation: <= UNE_MAX_LINES lines of <= UNE_MAX_CPL characters and
    <= UNE_MAX_DUR s; cuts preferably at pauses (> UNE_PAUSE s) and at the end of sentences.
    The display time is then extended (without overlapping) to reach UNE_MIN_DUR and, when there
    is room, to go below UNE_MAX_CPS. Segments without word timestamps are used as they are."""
    words = []
    for s in segments:
        if s.get("words"):
            words += [dict(w, word=w["word"].strip()) for w in s["words"] if w["word"].strip()]
        elif s["text"].strip():
            words.append({"word": s["text"].strip(), "start": s["start"], "end": s["end"]})
    if not words:
        return []
    fits = lambda t: len(_wrap(t, UNE_MAX_CPL)) <= UNE_MAX_LINES
    blocks, cur = [], []
    for w in words:
        if cur:
            text = " ".join(x["word"] for x in cur + [w])
            pause = w["start"] - cur[-1]["end"]
            sentence_end = cur[-1]["word"].endswith((".", "?", "!", "…"))
            if (not fits(text) or w["end"] - cur[0]["start"] > UNE_MAX_DUR or pause > UNE_PAUSE
                    or (sentence_end and len(" ".join(x["word"] for x in cur)) >= 15)):
                blocks.append(cur); cur = []
        cur.append(w)
    if cur:
        blocks.append(cur)
    subs = [{"start": b[0]["start"], "end": b[-1]["end"],
             "text": " ".join(x["word"] for x in b)} for b in blocks]
    for i, s in enumerate(subs):  # display-time adjustment
        limit = subs[i + 1]["start"] - UNE_MIN_GAP if i + 1 < len(subs) else s["end"] + UNE_MAX_DUR
        need = max(UNE_MIN_DUR, len(s["text"]) / UNE_MAX_CPS)
        target = min(s["start"] + min(need, UNE_MAX_DUR), limit)
        if target > s["end"]:
            s["end"] = target
        s["text"] = _balanced_lines(s["text"], UNE_MAX_CPL, UNE_MAX_LINES)
    return subs


SEGMENTERS = {"native": seg_native, "rules": seg_rules, "une": seg_une}


def apply_norm(subs: list, normalize) -> list:
    out = []
    for s in subs:
        lines = [normalize(l) for l in s["text"].split("\n")]
        t = "\n".join(l for l in lines if l.strip())
        if t.strip():
            out.append(dict(s, text=t))
    return out


def write_outputs(vdir: Path, model: str, decode_variant: str, segments: list, header: dict,
                  normalize) -> None:
    """Write the raw and the normalized variant (TXT + 3 SRT) of one decoding."""
    norm_variant = "norm" if decode_variant == "raw" else f"{decode_variant}+norm"
    for variant, norm_fn in [(decode_variant, None), (norm_variant, normalize)]:
        h = dict(header, Variante=variant, **{"Normalización": "sí (custom_normalizations)" if norm_fn else "no"})
        lines = [s["text"].strip() for s in segments]
        if norm_fn:
            lines = [norm_fn(l) for l in lines]
        h["Palabras transcritas"] = sum(len(l.split()) for l in lines)
        write_txt(vdir / f"{model}__{variant}.txt", h, lines)
        for seg_name, seg_fn in SEGMENTERS.items():
            subs = seg_fn(segments)
            if norm_fn:
                subs = apply_norm(subs, norm_fn)
            write_srt(vdir / f"{model}__{variant}__{seg_name}.srt", subs)


# ---------------------------------------------------------------------------
# ASR engines
# ---------------------------------------------------------------------------
def run_whisper(model_name, decode_variants, videos, out_dir, args, normalize, prompt, reglog):
    import ctranslate2
    from faster_whisper import WhisperModel
    pending = [(v, dv) for dv in decode_variants for v in videos
               if args.force or not (out_dir / f"VIDEO-{v}" / f"{model_name}__{dv}.txt").exists()]
    if not pending:
        log(f"{model_name}: all outputs exist, skipped"); return
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=args.cpu_threads)
    load_s = time.perf_counter() - t0
    log(f"{model_name}: loaded in {load_s:.1f} s ({CHECKPOINTS.get(model_name)})")
    first = load_wav(videos[min(videos)])
    list(model.transcribe(first[: int(WARMUP_SECONDS * SR)], language=LANGUAGE, beam_size=BEAM_SIZE)[0])
    for v, dv in pending:
        audio = load_wav(videos[v]); dur = len(audio) / SR
        init_prompt = prompt if dv == "prompt" else None
        set_seed(ctranslate2)
        t = time.perf_counter()
        seg_iter, info = model.transcribe(
            audio, language=LANGUAGE, beam_size=BEAM_SIZE, temperature=list(TEMPERATURES),
            word_timestamps=True, initial_prompt=init_prompt, vad_filter=False,
            condition_on_previous_text=True)
        segs = list(seg_iter)                       # transcription is lazy: it runs here
        elapsed = time.perf_counter() - t
        segments = [{"start": s.start, "end": s.end, "text": s.text, "temperature": s.temperature,
                     "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio,
                     "words": [{"word": w.word, "start": w.start, "end": w.end,
                                "probability": w.probability} for w in (s.words or [])]}
                    for s in segs]
        n_fallback = sum(1 for s in segments if s["temperature"] and s["temperature"] > 0)
        vdir = out_dir / f"VIDEO-{v}"; vdir.mkdir(parents=True, exist_ok=True)
        (vdir / f"{model_name}__{dv}.words.json").write_text(
            json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")
        header = {
            "Modelo": model_name, "Checkpoint": CHECKPOINTS.get(model_name, model_name),
            "Motor": f"faster-whisper {pv('faster-whisper')} / ctranslate2 {pv('ctranslate2')}",
            "Dispositivo": DEVICE.upper(), "Compute type": COMPUTE_TYPE, "Beam size": BEAM_SIZE,
            "Temperaturas": ",".join(str(x) for x in TEMPERATURES), "Semilla": SEED,
            "Segmentos con fallback": n_fallback,
            "Idioma": f"{LANGUAGE} (forzado)", "Prompt inicial": init_prompt or "ninguno",
            "Duración audio": f"{dur:.2f} segundos", "Carga del modelo": f"{load_s:.2f} segundos",
            "Tiempo transcurrido": f"{elapsed:.2f} segundos", "RTF": f"{elapsed / dur:.4f}",
        }
        write_outputs(vdir, model_name, dv, segments, header, normalize)
        reglog(model_name, dv, v, elapsed, dur, segments, n_fallback)
        log(f"{model_name:15s} {dv:7s} VIDEO-{v:<2d} {elapsed:7.1f} s  RTF={elapsed / dur:.3f}  "
            f"segs={len(segments)}  fallback={n_fallback}")
    del model


def run_wav2vec(videos, out_dir, args, normalize, reglog):
    import torch
    from wav2vec_transcriber import load_wav2vec2, DEFAULT_WAV2VEC2_MODEL_ID
    name = "wav2vec2"
    pending = [v for v in videos
               if args.force or not (out_dir / f"VIDEO-{v}" / f"{name}__raw.txt").exists()]
    if not pending:
        log("wav2vec2: all outputs exist, skipped"); return
    t0 = time.perf_counter()
    processor, model, _ = load_wav2vec2(DEFAULT_WAV2VEC2_MODEL_ID)
    load_s = time.perf_counter() - t0
    ptype = type(processor).__name__
    lm = ""
    if hasattr(processor, "decoder"):
        try:
            lmc = processor.decoder.model_container[processor.decoder._model_key]
            lm = f"KenLM (alpha={lmc.alpha}, beta={lmc.beta})"
        except Exception:
            lm = "KenLM"
    log(f"wav2vec2: {ptype} {lm} loaded in {load_s:.1f} s")
    first = load_wav(videos[min(videos)])[: int(WARMUP_SECONDS * SR)]
    with torch.no_grad():
        processor.decode(model(**processor(first, sampling_rate=SR, return_tensors="pt")).logits[0].numpy())
    ratio = model.config.inputs_to_logits_ratio / SR
    for v in pending:
        audio = load_wav(videos[v]); dur = len(audio) / SR
        t = time.perf_counter()
        with torch.no_grad():
            logits = model(**processor(audio, sampling_rate=SR, return_tensors="pt")).logits[0]
        try:
            dec = processor.decode(logits.numpy(), output_word_offsets=True)
        except Exception:
            dec = processor.decode(torch.argmax(logits, -1).numpy(), output_word_offsets=True)
        elapsed = time.perf_counter() - t
        words = [{"word": str(o["word"]).strip(), "start": round(o["start_offset"] * ratio, 2),
                  "end": round(o["end_offset"] * ratio, 2)} for o in (dec.word_offsets or [])
                 if str(o["word"]).strip()]
        # native segmentation of the application: blocks of <= 6 s or <= 12 words (src/wav2vec_transcriber.py)
        segments, cur = [], []
        for w in words:
            cur.append(w)
            if cur[-1]["end"] - cur[0]["start"] >= 6.0 or len(cur) >= 12:
                segments.append(cur); cur = []
        if cur:
            segments.append(cur)
        segments = [{"start": c[0]["start"], "end": c[-1]["end"],
                     "text": " ".join(x["word"] for x in c), "words": c} for c in segments]
        vdir = out_dir / f"VIDEO-{v}"; vdir.mkdir(parents=True, exist_ok=True)
        (vdir / f"{name}__raw.words.json").write_text(json.dumps(segments, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
        header = {"Modelo": name, "Checkpoint": DEFAULT_WAV2VEC2_MODEL_ID,
                  "Motor": f"transformers {pv('transformers')} / pyctcdecode {pv('pyctcdecode')}",
                  "Decodificador": f"{ptype} {lm}".strip(), "Dispositivo": "CPU",
                  "Idioma": "es (modelo monolingüe)", "Prompt inicial": "no soportado",
                  "Duración audio": f"{dur:.2f} segundos", "Carga del modelo": f"{load_s:.2f} segundos",
                  "Tiempo transcurrido": f"{elapsed:.2f} segundos", "RTF": f"{elapsed / dur:.4f}"}
        write_outputs(vdir, name, "raw", segments, header, normalize)
        reglog(name, "raw", v, elapsed, dur, segments, 0)
        log(f"{name:15s} raw     VIDEO-{v:<2d} {elapsed:7.1f} s  RTF={elapsed / dur:.3f}")


def check_determinism(videos, out_dir, args, prompt):
    """Transcribe the first video twice with 'small' and compare the text."""
    import ctranslate2
    from faster_whisper import WhisperModel
    m = WhisperModel("small", device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=args.cpu_threads)
    audio = load_wav(videos[min(videos)]); texts = []
    for _ in range(2):
        set_seed(ctranslate2)
        segs, _ = m.transcribe(audio, language=LANGUAGE, beam_size=BEAM_SIZE,
                               temperature=list(TEMPERATURES), word_timestamps=True)
        texts.append(" ".join(s.text for s in segs))
    ok = texts[0] == texts[1]
    log(f"Determinism (small, VIDEO-{min(videos)}, 2 runs): {'IDENTICAL' if ok else 'DIFFERENT'}")
    return ok


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=WHISPER_MODELS + ["wav2vec2"])
    ap.add_argument("--prompt-models", nargs="*", default=PROMPT_MODELS)
    ap.add_argument("--only-videos", nargs="*", type=int)
    ap.add_argument("--cpu-threads", type=int, default=0, help="0 = CTranslate2 default")
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--resegment-only", action="store_true",
                    help="regenerate TXT/SRT from the .words.json files without transcribing again")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(SRC_DIR))
    from custom_normalizations import normalize_text
    from whisper_prompts import DEFAULT_PROMPT

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not installed or not in the PATH")
    srcs = {video_id(p.name): p for p in sorted(args.videos.iterdir())
            if video_id(p.name) and not p.name.startswith(".")}
    if args.only_videos:
        srcs = {v: p for v, p in srcs.items() if v in args.only_videos}
    if not srcs:
        sys.exit(f"No files named 'VIDEO-N' in {args.videos}")
    args.out.mkdir(parents=True, exist_ok=True)

    plan = [(m, ["raw"] + (["prompt"] if m in args.prompt_models else []))
            for m in args.models if m != "wav2vec2"]
    log(f"Videos: {sorted(srcs)}")
    for m, dv in plan:
        log(f"  {m:15s} decodings: {dv}")
    if "wav2vec2" in args.models:
        log("  wav2vec2        decodings: ['raw']")
    log(f"Prompt: {DEFAULT_PROMPT}")
    if args.dry_run:
        return

    if args.resegment_only:
        for vdir in sorted(args.out.glob("VIDEO-*")):
            for js in vdir.glob("*.words.json"):
                model, dv = js.name.replace(".words.json", "").split("__")
                txt = vdir / f"{model}__{dv}.txt"
                header = {}
                if txt.exists():
                    for line in txt.read_text(encoding="utf-8").splitlines():
                        if not line.strip(): break
                        k, _, val = line.partition(":"); header[k.strip()] = val.strip()
                header.pop("Variante", None); header.pop("Normalización", None)
                write_outputs(vdir, model, dv, json.loads(js.read_text(encoding="utf-8")), header,
                              normalize_text)
        log("Re-segmentation finished"); return

    wav_dir = args.out / "_audio_16k"
    videos = {v: to_wav16k(p, wav_dir / f"VIDEO-{v}.wav") for v, p in srcs.items()}
    env = environment_info(args)
    env["prompt_inicial"] = DEFAULT_PROMPT
    env["duraciones_s"] = {v: round(len(load_wav(p)) / SR, 2) for v, p in videos.items()}
    (args.out / "entorno.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Hardware: {env['chip']} | RAM {env['ram_GB']} GB | cores {env['nucleos_fisicos']}")

    log_path = args.out / "registro_ejecucion.csv"
    new = not log_path.exists()
    fh = log_path.open("a", newline="", encoding="utf-8"); wr = csv.writer(fh)
    if new:
        wr.writerow(["fecha", "modelo", "decodificacion", "video", "tiempo_s", "duracion_s", "rtf",
                     "segmentos", "palabras", "segmentos_fallback"])

    def reglog(model, dv, v, elapsed, dur, segments, nfb):
        wr.writerow([datetime.now().isoformat(timespec="seconds"), model, dv, v, f"{elapsed:.3f}",
                     f"{dur:.3f}", f"{elapsed / dur:.5f}", len(segments),
                     sum(len(s["text"].split()) for s in segments), nfb]); fh.flush()

    if args.check_determinism:
        check_determinism(videos, args.out, args, DEFAULT_PROMPT)
    t_all = time.perf_counter()
    for m, dv in plan:
        run_whisper(m, dv, videos, args.out, args, normalize_text, DEFAULT_PROMPT, reglog)
    if "wav2vec2" in args.models:
        run_wav2vec(videos, args.out, args, normalize_text, reglog)
    fh.close()
    log(f"Finished in {(time.perf_counter() - t_all) / 60:.1f} min. Outputs in {args.out}")


if __name__ == "__main__":
    main()
