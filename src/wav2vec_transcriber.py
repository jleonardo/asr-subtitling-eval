"""Wav2Vec 2.0 loader used by the transcription application (jonatasgrosman/wav2vec2-large-xlsr-53-spanish).

The checkpoint ships a 5-gram KenLM language model, so AutoProcessor returns a Wav2Vec2ProcessorWithLM
(pyctcdecode beam search). scripts/rerun_controlled.py only uses load_wav2vec2() and DEFAULT_WAV2VEC2_MODEL_ID.
"""
from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from custom_normalizations import normalize_text


DEFAULT_WAV2VEC2_MODEL_ID = os.getenv(
    "WAV2VEC2_MODEL_ID",
    "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
)

WAV2VEC2_CAPABILITIES: Dict[str, bool] = {
    "basic_transcription": True,
    "comparison": True,
    "local_inference": True,
    "streaming": False,
    "realtime": False,
    "translation": False,
    "initial_prompt": False,
    "language_detection": False,
    "language_probability": False,
    "word_timestamps": True,
}


@dataclass
class Wav2VecWord:
    word: str
    start: float
    end: float


@dataclass
class Wav2VecSegment:
    start: float
    end: float
    text: str
    words: Optional[List[Wav2VecWord]] = None


@dataclass
class Wav2VecInfo:
    language: str
    language_probability: Optional[float]
    model_id: str
    capabilities: Dict[str, bool]


_MODEL_CACHE: Dict[str, object] = {}
_PROCESSOR_CACHE: Dict[str, object] = {}


def wav2vec_is_available() -> Tuple[bool, Optional[str]]:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCTC, AutoProcessor  # noqa: F401
        import pyctcdecode  # noqa: F401
        import kenlm  # noqa: F401
        return True, None
    except Exception as exc:
        return False, str(exc)


def _load_audio_array(input_path: Union[str, Path]) -> np.ndarray:
    with wave.open(str(input_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_audio = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError("Wav2Vec2 requiere audio PCM de 16 bits")

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = audio / 32768.0

    if sample_rate != 16000:
        source_index = np.arange(audio.shape[0], dtype=np.float32)
        target_size = int(round(audio.shape[0] * (16000.0 / float(sample_rate))))
        target_index = np.linspace(0, max(audio.shape[0] - 1, 0), num=max(target_size, 1), dtype=np.float32)
        audio = np.interp(target_index, source_index, audio).astype(np.float32)

    return audio.astype(np.float32)


def _build_segments_from_words(
    text: str,
    words: List[Wav2VecWord],
    max_duration: float = 6.0,
    max_words: int = 12,
) -> List[Wav2VecSegment]:
    if not words:
        cleaned = normalize_text((text or "").strip())
        return [Wav2VecSegment(start=0.0, end=0.0, text=cleaned, words=None)] if cleaned else []

    segments: List[Wav2VecSegment] = []
    current_words: List[Wav2VecWord] = []
    current_start = words[0].start

    for word in words:
        if not current_words:
            current_start = word.start

        current_words.append(word)
        current_duration = word.end - current_start

        if current_duration >= max_duration or len(current_words) >= max_words:
            segment_text = normalize_text(" ".join(item.word for item in current_words).strip())
            if segment_text:
                segments.append(
                    Wav2VecSegment(
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        text=segment_text,
                        words=current_words.copy(),
                    )
                )
            current_words = []

    if current_words:
        segment_text = normalize_text(" ".join(item.word for item in current_words).strip())
        if segment_text:
            segments.append(
                Wav2VecSegment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=segment_text,
                    words=current_words.copy(),
                )
            )

    return segments


def load_wav2vec2(model_id: str = DEFAULT_WAV2VEC2_MODEL_ID):
    available, error = wav2vec_is_available()
    if not available:
        raise RuntimeError(
            "Wav2Vec2 no disponible. Instala torch y transformers. "
            f"Detalle: {error}"
        )

    if model_id not in _MODEL_CACHE or model_id not in _PROCESSOR_CACHE:
        from transformers import AutoModelForCTC, AutoProcessor

        _PROCESSOR_CACHE[model_id] = AutoProcessor.from_pretrained(model_id)
        _MODEL_CACHE[model_id] = AutoModelForCTC.from_pretrained(model_id)
        _MODEL_CACHE[model_id].eval()

    return _PROCESSOR_CACHE[model_id], _MODEL_CACHE[model_id], "cpu"


def transcribe_file_wav2vec(
    input_path: Union[str, Path],
    model_id: str = DEFAULT_WAV2VEC2_MODEL_ID,
) -> Tuple[List[Wav2VecSegment], Wav2VecInfo, float]:
    processor, model, used_device = load_wav2vec2(model_id=model_id)
    audio = _load_audio_array(input_path)

    import torch

    started = time.perf_counter()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]

    logits_np = logits.cpu().numpy()
    predicted_ids = torch.argmax(logits, dim=-1)

    # Wav2Vec2ProcessorWithLM expects 2D logits (time, vocab),
    # while the base processor decode path expects token IDs.
    try:
        decoded = processor.decode(logits_np, output_word_offsets=True)
    except Exception:
        decoded = processor.decode(predicted_ids.cpu().numpy(), output_word_offsets=True)
    elapsed_seconds = time.perf_counter() - started

    time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
    word_offsets = getattr(decoded, "word_offsets", None) or []

    words: List[Wav2VecWord] = []
    for item in word_offsets:
        word = normalize_text(str(item.get("word", "")).strip())
        if not word:
            continue
        words.append(
            Wav2VecWord(
                word=word,
                start=round(float(item.get("start_offset", 0)) * time_offset, 2),
                end=round(float(item.get("end_offset", 0)) * time_offset, 2),
            )
        )

    text = normalize_text(getattr(decoded, "text", str(decoded)).strip())
    segments = _build_segments_from_words(text=text, words=words)
    if not segments and text:
        duration = words[-1].end if words else 0.0
        segments = [Wav2VecSegment(start=0.0, end=duration, text=text, words=words or None)]

    info = Wav2VecInfo(
        language="es",
        language_probability=None,
        model_id=model_id,
        capabilities=WAV2VEC2_CAPABILITIES.copy(),
    )
    return segments, info, elapsed_seconds


def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds >= 1000:
        milliseconds -= 1000
        whole_seconds += 1
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def write_wav2vec_outputs(
    segments: List[Wav2VecSegment],
    info: Wav2VecInfo,
    elapsed_seconds: float,
    output_dir: Path,
    base_name: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{base_name}_transcription.txt"
    srt_path = output_dir / f"{base_name}_transcription.srt"

    word_count = sum(len(segment.text.split()) for segment in segments)
    probability_text = "n/d"

    with txt_path.open("w", encoding="utf-8") as txt_file:
        txt_file.write("Modelo: Wav2Vec2\n")
        txt_file.write(f"Modelo HF: {info.model_id}\n")
        txt_file.write(f"Idioma: {info.language} (Probabilidad: {probability_text})\n")
        txt_file.write(f"Palabras transcritas: {word_count}\n")
        txt_file.write(f"Tiempo transcurrido: {elapsed_seconds:.2f} segundos\n")
        txt_file.write("\n")
        txt_file.write("\n".join(segment.text for segment in segments))

    with srt_path.open("w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(segments, start=1):
            start_time = seconds_to_srt_time(segment.start)
            end_time = seconds_to_srt_time(segment.end)
            srt_file.write(
                f"{index}\n{start_time} --> {end_time}\n<font color=\"white\"> {segment.text}</font>\n\n"
            )

    return txt_path, srt_path