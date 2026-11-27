"""
check_whisper.py - Shows which checkpoint faster-whisper uses for each model alias and its number of parameters.

Usage (from the repository root):
    python scripts/check_whisper.py                      # models of the paper
    python scripts/check_whisper.py large large-v3       # specific aliases
    python scripts/check_whisper.py --load               # also loads each model on CPU/int8

For each alias it reports:
  1. CTranslate2 repository the alias points to in faster-whisper.
  2. Local cache folder and size of model.bin (if already downloaded).
  3. Architecture and number of parameters, computed from the configuration of the original
     OpenAI model (no weights are downloaded: only config.json).
  4. Parameter estimate from model.bin (float16 weights = 2 bytes).
  5. (--load) n_mels and multilingual flag read from the loaded model.

This script was used to verify that, in faster-whisper, the alias "large" resolves to large-v3.
"""

import argparse
import os
import sys

import faster_whisper
from faster_whisper.utils import _MODELS, download_model

PAPER_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "large"]


def openai_repo_from_ct2(repo_id: str) -> str:
    """Systran/faster-whisper-large-v3 -> openai/whisper-large-v3
    mobiuslabsgmbh/faster-whisper-large-v3-turbo -> openai/whisper-large-v3-turbo"""
    name = repo_id.split("/")[-1]
    if "distil" in name:
        return "distil-whisper/" + name.replace("faster-distil-whisper-", "distil-")
    return "openai/" + name.replace("faster-whisper-", "whisper-")


def local_snapshot(repo_or_alias: str):
    try:
        return download_model(repo_or_alias, local_files_only=True)
    except Exception:
        return None


def count_params(openai_repo: str):
    """Instantiate the model on the 'meta' device (no memory, no weights) and count its parameters."""
    import torch
    from transformers import AutoConfig, WhisperForConditionalGeneration

    cfg = AutoConfig.from_pretrained(openai_repo)
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(cfg)
    total = sum(p.numel() for p in model.parameters())  # shared parameters are counted once
    enc = sum(p.numel() for p in model.model.encoder.parameters())
    dec = sum(p.numel() for p in model.model.decoder.parameters())
    return cfg, total, enc, dec


def fmt_m(n):
    return f"{n / 1e6:,.1f} M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=PAPER_MODELS)
    ap.add_argument("--load", action="store_true",
                    help="load each model with faster-whisper (CPU, int8) to read n_mels")
    args = ap.parse_args()

    print(f"faster-whisper {faster_whisper.__version__}")
    try:
        import ctranslate2
        print(f"ctranslate2    {ctranslate2.__version__}")
    except Exception:
        pass
    print("=" * 78)

    for alias in args.models:
        repo = _MODELS.get(alias, alias)
        print(f"\nAlias:                 {alias}")
        print(f"Checkpoint (CT2):      {repo}")

        path = local_snapshot(alias)
        if path:
            print(f"Local cache:           {path}")
            mb = os.path.join(path, "model.bin")
            if os.path.exists(mb):
                size = os.path.getsize(mb)
                print(f"model.bin:             {size / 1e9:.2f} GB  "
                      f"(≈ {fmt_m(size / 2)} parameters if float16)")
        else:
            print("Local cache:           not downloaded on this machine")

        oa = openai_repo_from_ct2(repo)
        try:
            cfg, total, enc, dec = count_params(oa)
            print(f"Original model:        {oa}")
            print(f"Architecture:          encoder {cfg.encoder_layers} layers, "
                  f"decoder {cfg.decoder_layers} layers, d_model {cfg.d_model}, "
                  f"n_mels {cfg.num_mel_bins}")
            print(f"Total parameters:      {fmt_m(total)}  "
                  f"(encoder {fmt_m(enc)}, decoder {fmt_m(dec)})")
        except Exception as e:
            print(f"Parameters:            could not be computed ({type(e).__name__}: {e})")

        if args.load:
            try:
                from faster_whisper import WhisperModel
                m = WhisperModel(alias, device="cpu", compute_type="int8")
                ct2 = m.model
                print(f"Loaded (CPU, int8):    n_mels={ct2.n_mels}, "
                      f"multilingual={ct2.is_multilingual}")
                del m
            except Exception as e:
                print(f"Loading:               failed ({type(e).__name__}: {e})")

    print("\n" + "=" * 78)
    same = {}
    for a in args.models:
        same.setdefault(_MODELS.get(a, a), []).append(a)
    for repo, aliases in same.items():
        if len(aliases) > 1:
            print(f"NOTE: {', '.join(aliases)} point to the SAME checkpoint ({repo})")


if __name__ == "__main__":
    sys.exit(main())
