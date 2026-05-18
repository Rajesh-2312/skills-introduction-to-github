import queue
import threading
from typing import Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class Speaker:
    """Background-thread TTS so speech never blocks the camera loop."""

    def __init__(self, enabled: bool = True, voice_id: Optional[str] = None, rate: int = 175):
        self._available = enabled and pyttsx3 is not None
        self._muted = False
        self._voice_id = voice_id
        self._rate = rate
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        if self._available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        elif enabled and pyttsx3 is None:
            print("[tts] pyttsx3 not installed — install it or pass --no-tts to silence this message.")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def muted(self) -> bool:
        return self._muted

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        return self._muted

    def speak(self, text: str) -> None:
        if not self._available or self._muted or not text:
            return
        # Drop any pending utterance so the latest object wins.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text)

    def shutdown(self) -> None:
        if self._available:
            self._queue.put(None)

    def _run(self) -> None:
        try:
            engine = pyttsx3.init()
        except Exception as e:
            print(f"[tts] could not initialize engine: {e}")
            self._available = False
            return

        if self._voice_id:
            engine.setProperty("voice", self._voice_id)
        engine.setProperty("rate", self._rate)

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[tts] error: {e}")


def summarize_for_speech(title: str, summary: str, max_chars: int = 400) -> str:
    """Take a Wikipedia title+summary and produce a TTS-friendly utterance."""
    body = summary.strip()
    if len(body) > max_chars:
        cut = body.rfind(". ", 0, max_chars)
        body = body[: cut + 1] if cut > 0 else body[:max_chars] + "..."
    return f"{title}. {body}" if title else body
