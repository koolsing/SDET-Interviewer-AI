"""
tts.py — Text-to-speech via Piper binary (subprocess)
Pipes text → Piper → raw PCM → plays via sounddevice.
"""
import subprocess
import threading
import numpy as np
import sounddevice as sd

from config import PIPER_BIN, VOICE_MODEL, PIPER_SAMPLE_RATE, PIPER_CHANNELS


def speak(text: str, blocking: bool = True) -> None:
    """
    Convert text to speech using Piper and play immediately.
    blocking=True waits for playback to finish before returning.
    """
    if not text or not text.strip():
        return

    # Sanitise: remove markdown symbols that sound odd when spoken
    clean = (
        text
        .replace("**", "")
        .replace("*", "")
        .replace("`", "")
        .replace("#", "")
        .replace(">", "")
        .replace("\n\n", ". ")
        .replace("\n", " ")
        .strip()
    )

    def _play():
        try:
            proc = subprocess.Popen(
                [PIPER_BIN, "--model", VOICE_MODEL, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            raw_audio, _ = proc.communicate(input=clean.encode("utf-8"))

            if not raw_audio:
                return

            # Piper outputs 16-bit signed PCM
            audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, samplerate=PIPER_SAMPLE_RATE)
            sd.wait()
        except FileNotFoundError:
            # Piper not installed yet — fall back to silent no-op
            print(f"[TTS unavailable] {clean}")
        except Exception as e:
            print(f"[TTS error] {e}")

    if blocking:
        _play()
    else:
        t = threading.Thread(target=_play, daemon=True)
        t.start()


def piper_available() -> bool:
    """Return True if the Piper binary and voice model exist."""
    import os
    return os.path.isfile(PIPER_BIN) and os.path.isfile(VOICE_MODEL)
