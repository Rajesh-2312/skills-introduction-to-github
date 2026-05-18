import queue
import threading
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None


SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480 samples
SILENCE_FRAMES_END = 20      # ~600 ms of silence ends an utterance
MAX_UTTERANCE_FRAMES = 400   # hard cap ~12 s
MIN_UTTERANCE_FRAMES = 5     # ignore <150 ms blips


def parse_voice_command(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Map a free-form transcript to (command, arg). 'unknown' means no match."""
    t = text.lower().strip().strip(".?!,")
    if not t:
        return None, None

    if t.startswith("ask ") or t.startswith("question "):
        return "ask", text.split(maxsplit=1)[1] if " " in text else ""
    if any(p in t for p in ("what is this", "whats this", "what's this",
                            "tell me about this", "describe this", "show details", "details")):
        return "details", None
    if t in ("next", "next one", "go next"):
        return "next", None
    if t in ("previous", "prev", "back", "previous one", "go back"):
        return "prev", None
    if "mute" in t or "be quiet" in t:
        return "mute", None
    if "repeat" in t or "again" in t or "replay" in t:
        return "replay", None
    if t in ("clear", "hide", "close panel"):
        return "clear", None
    if t in ("quit", "exit", "stop", "close", "shutdown"):
        return "quit", None
    return "unknown", text


class VoiceListener:
    """Always-on voice command listener (mic + VAD + faster-whisper)."""

    def __init__(
        self,
        callback: Callable[[str, Optional[str]], None],
        model_size: str = "base",
        language: str = "en",
        vad_aggressiveness: int = 2,
    ):
        self.callback = callback
        self.available = sd is not None and WhisperModel is not None and webrtcvad is not None
        if not self.available:
            missing = [
                n for n, m in [("sounddevice", sd), ("faster_whisper", WhisperModel),
                               ("webrtcvad", webrtcvad)]
                if m is None
            ]
            print(f"[voice] disabled - missing packages: {', '.join(missing)}")
            return
        self._model_size = model_size
        self._language = language
        self._vad_aggressiveness = vad_aggressiveness
        self._stop = threading.Event()
        self._audio_q: "queue.Queue[bytes]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        if self.available:
            self._stop.set()

    def _run(self) -> None:
        print(f"[voice] loading faster-whisper '{self._model_size}' (first run downloads weights)...")
        try:
            model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        except Exception as e:
            print(f"[voice] could not load model: {e}")
            self.available = False
            return
        vad = webrtcvad.Vad(self._vad_aggressiveness)

        def audio_cb(indata, _frames, _ts, status):
            if status:
                print(f"[voice] {status}")
            self._audio_q.put(indata.tobytes())

        try:
            stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype="int16",
                blocksize=FRAME_SIZE,
                callback=audio_cb,
            )
            stream.start()
        except Exception as e:
            print(f"[voice] could not open microphone: {e}")
            self.available = False
            return

        print("[voice] listening - try saying 'what is this'")
        in_speech = False
        silence_count = 0
        utterance: list[bytes] = []

        try:
            while not self._stop.is_set():
                try:
                    pcm = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    is_speech = vad.is_speech(pcm, SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if is_speech:
                    in_speech = True
                    silence_count = 0
                    utterance.append(pcm)
                elif in_speech:
                    silence_count += 1
                    utterance.append(pcm)

                if len(utterance) >= MAX_UTTERANCE_FRAMES or (
                    in_speech and silence_count >= SILENCE_FRAMES_END
                ):
                    if len(utterance) >= MIN_UTTERANCE_FRAMES:
                        self._transcribe(model, b"".join(utterance))
                    utterance.clear()
                    in_speech = False
                    silence_count = 0
        finally:
            stream.stop()
            stream.close()

    def _transcribe(self, model, raw: bytes) -> None:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            segments, _ = model.transcribe(audio, language=self._language, vad_filter=False)
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            print(f"[voice] transcription error: {e}")
            return
        if not text:
            return
        cmd, arg = parse_voice_command(text)
        print(f"[voice] heard: {text!r} -> ({cmd}, {arg!r})")
        if cmd is not None:
            self.callback(cmd, arg if arg is not None else text)
