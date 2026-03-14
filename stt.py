"""
stt.py — Speech-to-text using faster-whisper + sounddevice
Records from microphone until silence detected, then transcribes.
"""
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import threading

from config import (
    WHISPER_MODEL, WHISPER_DEVICE,
    SAMPLE_RATE, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_RECORD_SECONDS,
)

_model: WhisperModel | None = None

def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="int8")
    return _model

# ── Recording ─────────────────────────────────────────────────────────────────

def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

def record_until_silence(
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
    max_seconds: float = MAX_RECORD_SECONDS,
    status_callback=None,
    stop_event: threading.Event = None,
) -> np.ndarray:
    """
    Record from default microphone.
    Stops when silence_duration seconds of silence detected, or max_seconds reached.
    Returns float32 mono numpy array at sample_rate Hz.
    """
    chunk_ms = 100                          # process in 100 ms chunks
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    max_chunks = int(max_seconds * 1000 / chunk_ms)

    audio_chunks = []
    silent_chunks = 0
    silence_chunks_needed = int(silence_duration * 1000 / chunk_ms)
    started_speaking = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        for _ in range(max_chunks):
            if stop_event and stop_event.is_set():
                break

            data, _ = stream.read(chunk_samples)
            audio_chunks.append(data.copy())
            rms = _rms(data)

            if rms > silence_threshold * 32768:  # int16 scale
                started_speaking = True
                silent_chunks = 0
            else:
                if started_speaking:
                    silent_chunks += 1
                    if silent_chunks >= silence_chunks_needed:
                        break

    if not audio_chunks:
        return np.array([], dtype=np.float32)

    raw = np.concatenate(audio_chunks, axis=0).flatten()
    return raw.astype(np.float32) / 32768.0   # normalise to [-1, 1]


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe a float32 numpy audio array. Returns the text."""
    if audio is None or len(audio) == 0:
        return ""
    model = _get_model()
    segments, _ = model.transcribe(audio, beam_size=5, language="en")
    return " ".join(s.text.strip() for s in segments).strip()


def listen_and_transcribe(status_callback=None, stop_event: threading.Event = None) -> str:
    """
    Record from mic → transcribe → return text.
    status_callback(msg) called with status strings if provided.
    """
    if status_callback:
        status_callback("Listening…  (speak now, pause 7 s to finish)")
    audio = record_until_silence(stop_event=stop_event)
    if status_callback:
        status_callback("Transcribing…")
    return transcribe(audio)


# ── Smoke test (no mic needed) ────────────────────────────────────────────────

def test_whisper():
    """Sanity-check: load model and transcribe a silent 1-second clip."""
    model = _get_model()
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    segments, _ = model.transcribe(silent, beam_size=1, language="en")
    text = " ".join(s.text for s in segments)
    print(f"Whisper smoke test OK. Transcribed silence: '{text}'")
