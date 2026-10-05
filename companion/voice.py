"""Near-real-time voice for the companion via a local Fish Audio S2 server.

Design: the LLM streams text, sentences are completed mid-stream and each one
is synthesized in a background worker while the previous sentence plays. Voice
consistency across a reply comes from chained references: each sentence reuses
the previous sentence's audio+text as its voice reference.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

try:
    import ormsgpack
except ImportError:  # pragma: no cover
    ormsgpack = None


# Inline emotion tags understood by Fish Audio S2. Free-form tags also work,
# but the model is instructed to stay near this list for reliability.
VOICE_TAG_GUIDE = """You may add up to 3 emotion tags inline using [brackets] to make speech sound natural (e.g. [excited] [sigh] [soft laugh] [short pause] [whisper] [surprised] [sad] [delight] [chuckle] [low voice] [laughing tone] [exhale]). Use them sparingly, only where a human would show real feeling. Never invent other bracketed text."""

TAG_RE = re.compile(r"\[[^\][\n]{1,64}\]")
WS_RE = re.compile(r"\s+")
# Sentence ends, but never inside [...] (tags carry no end punctuation).
BOUNDARY_RE = re.compile(r"[.!?…]+[\"'\"\u201d\u2019)\]]*\s+")
ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "eg", "ie", "jr", "sr", "no", "ft", "al"}

VOICE_GAP_SECONDS = 0.5


def strip_tags(text: str) -> str:
    cleaned = TAG_RE.sub(" ", text)
    cleaned = WS_RE.sub(" ", cleaned).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", cleaned)


def _has_speech(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", TAG_RE.sub("", text)))


def _split_points(text: str) -> list[int]:
    points = []
    for match in BOUNDARY_RE.finditer(text):
        before = text[: match.start()]
        word = re.search(r"([A-Za-z.]{1,6})$", before)
        token = (word.group(1) if word else "").lower().replace(".", "")
        if token in ABBREV:
            continue
        points.append(match.end())
    return points


def split_sentences(text: str, max_len: int = 450) -> list[str]:
    """Split whole text into speakable sentences (used when reply arrives whole)."""
    streamer = SentenceStreamer()
    sentences = streamer.feed(text)
    sentences.extend(streamer.flush())
    return sentences


class SentenceStreamer:
    """Incrementally carve complete sentences out of a growing text stream."""

    def __init__(self, min_len: int = 24, max_len: int = 450):
        self.buf = ""
        self.min_len = min_len
        self.max_len = max_len

    def feed(self, chunk: str) -> list[str]:
        self.buf += chunk
        return self._drain(final=False)

    def flush(self) -> list[str]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[str]:
        points = _split_points(self.buf)
        if not points:
            if final and _has_speech(self.buf):
                piece, self.buf = self.buf.strip(), ""
                return self._fit(piece)
            return []
        if final:
            complete, self.buf = self.buf, ""
        else:
            complete, self.buf = self.buf[: points[-1]], self.buf[points[-1] :]
        # Every piece here ends with terminal punctuation, so even short ones
        # ("Hey!") are complete speakable units: emit immediately for true
        # progressive speech instead of batching everything to flush().
        pieces = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", complete) if part.strip()]
        out: list[str] = []
        for piece in pieces:
            if _has_speech(piece):
                out.extend(self._fit(piece))
        return out

    def _fit(self, piece: str) -> list[str]:
        if len(piece) <= self.max_len:
            return [piece]
        out: list[str] = []
        rest = piece
        while len(rest) > self.max_len:
            cut = max(
                rest.rfind(",", 0, self.max_len),
                rest.rfind(";", 0, self.max_len),
                rest.rfind(" \u2014 ", 0, self.max_len),
                rest.rfind(" - ", 0, self.max_len),
            )
            if cut < self.max_len // 2:
                cut = rest.rfind(" ", 0, self.max_len)
            if cut <= 0:
                break
            out.append(rest[: cut + 1].strip())
            rest = rest[cut + 1 :].strip()
        if rest and _has_speech(rest):
            out.append(rest)
        return out


def _concat_track(wavs: list[bytes], gap_seconds: float = 0.5) -> bytes:
    """Join clips into one continuous track with a fixed silence gap between
    each sentence, so playback is strictly sequential with real pauses."""
    import io
    import wave

    frames = b""
    params = None
    for data in wavs:
        with wave.open(io.BytesIO(data), "rb") as clip:
            clip_params = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate())
            if params is None:
                params = clip_params
            elif clip_params != params:
                raise RuntimeError("Mismatched TTS audio params, cannot concatenate")
            frames += clip.readframes(clip.getnframes())
            gap_frames = int(clip.getframerate() * gap_seconds)
            frames += b"\x00" * gap_frames * clip.getnchannels() * clip.getsampwidth()
    if params is None:
        raise RuntimeError("No audio to concatenate")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(params[0])
        out.setsampwidth(params[1])
        out.setframerate(params[2])
        out.writeframes(frames)
    return buf.getvalue()


class VoiceService:
    """Speaks reply sentences through the local fish-speech HTTP server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        reference_id: str | None = None,
        timeout: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.reference_id = reference_id
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-tts")
        self._last_audio: dict[tuple[str, int], tuple[str, bytes]] = {}
        self._key: tuple[str, int] | None = None
        self._turn_lock = threading.Lock()
        self._chain_lock = threading.Lock()
        self._server_start_attempted = False
        self._turn_clips: dict[tuple[str, int], list[tuple[int, bytes]]] = {}
        self._turn_complete: set[tuple[str, int]] = set()
        self._turn_in_flight: dict[tuple[str, int], int] = {}
        self._turn_delivered: set[tuple[str, int]] = set()

    def health(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url}/v1/health", timeout=5)
            if response.status_code == 200:
                return True, "Voice server is ready."
            return False, f"Voice server answered {response.status_code}."
        except Exception as error:
            return False, f"Voice server unreachable at {self.base_url}: {error}"

    def ensure_server(self, project_root: Path) -> tuple[bool, str]:
        """Start the sidecar TTS server once if it is not already running."""
        healthy, _ = self.health()
        if healthy:
            return True, "Voice server is ready."
        if self._server_start_attempted:
            return False, "Voice server is still starting (first start takes a few minutes)."
        python_exe = project_root / "voice_env" / "Scripts" / "python.exe"
        if not python_exe.exists():
            return False, "voice_env is missing; see README voice setup."
        env = dict(os.environ)
        env["FISH_MAX_SEQ_LEN"] = "4096"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        log_path = project_root / "voice_server.log"
        try:
            with open(log_path, "ab") as log_handle:
                subprocess.Popen(
                    [
                        str(python_exe),
                        "tools/api_server.py",
                        "--llama-checkpoint-path",
                        str(project_root / "Models" / "s2-pro-int8"),
                        "--decoder-checkpoint-path",
                        str(project_root / "Models" / "s2-pro-int8" / "codec.pth"),
                        "--listen",
                        "0.0.0.0:8080",
                    ],
                    cwd=str(project_root / "vendor" / "fish-speech"),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                )
        except Exception as error:
            return False, f"Could not start voice server: {error}"
        self._server_start_attempted = True
        return False, "Voice server is starting (first start takes a few minutes)."

    def new_turn(self, run_id: str, turn: int) -> None:
        """Mark a turn active; older turns' pending speech is dropped silently."""
        with self._turn_lock:
            self._key = (run_id, turn)
            self._last_audio = {key: value for key, value in self._last_audio.items() if key == self._key}
            self._turn_clips = {key: value for key, value in self._turn_clips.items() if key == self._key}
            self._turn_complete = {key for key in self._turn_complete if key == self._key}
            self._turn_in_flight = {key: value for key, value in self._turn_in_flight.items() if key == self._key}
            self._turn_delivered = {key for key in self._turn_delivered if key == self._key}

    def submit(self, index: int, sentence: str) -> None:
        if not _has_speech(sentence):
            return
        with self._turn_lock:
            key = self._key
            if key is None:
                return
            self._turn_in_flight[key] = self._turn_in_flight.get(key, 0) + 1
        try:
            self._executor.submit(self._synthesize_one, key, index, sentence)
        except Exception:
            with self._turn_lock:
                self._turn_in_flight[key] = max(0, self._turn_in_flight.get(key, 0) - 1)
            raise

    def finish_turn(self) -> None:
        """Signal that no more sentences will be submitted for the active turn."""
        with self._turn_lock:
            key = self._key
            if key is not None:
                self._turn_complete.add(key)

    def pop_ready(self) -> list[dict[str, Any]]:
        """Return one combined track per finished turn (single playback each)."""
        ready: list[dict[str, Any]] = []
        with self._turn_lock:
            for key in list(self._turn_complete):
                if key in self._turn_delivered:
                    continue
                if self._turn_in_flight.get(key, 0) > 0:
                    continue
                clips = sorted(self._turn_clips.get(key, []))
                if not clips:
                    self._turn_delivered.add(key)
                    continue
                try:
                    track = _concat_track([audio for _, audio in clips])
                except Exception as error:
                    print(f"[voice] concat failed, skipping turn audio: {error}")
                    self._turn_delivered.add(key)
                    continue
                self._turn_delivered.add(key)
                ready.append({"run": key[0], "turn": key[1], "audio": track})
        return ready

    def _synthesize_one(self, key: tuple[str, int], index: int, sentence: str) -> None:
        try:
            with self._turn_lock:
                if key != self._key:
                    return
            with self._chain_lock:
                previous = self._last_audio.get(key)
            references = []
            if previous is not None:
                prev_text, prev_audio = previous
                references.append({"audio": prev_audio, "text": prev_text})
            audio = self._request(sentence, references)
            with self._turn_lock:
                if key != self._key:
                    return
                self._last_audio[key] = (sentence, audio)
                self._turn_clips.setdefault(key, []).append((index, audio))
        except Exception as error:
            print(f"[voice] synthesis failed, skipping sentence: {error}")
        finally:
            with self._turn_lock:
                self._turn_in_flight[key] = max(0, self._turn_in_flight.get(key, 0) - 1)

    def _request(self, text: str, references: list[dict[str, Any]]) -> bytes:
        if ormsgpack is None:
            raise RuntimeError("ormsgpack is not installed; run pip install -r requirements.txt")
        max_tokens = max(384, min(1024, len(text) * 4))
        payload = {
            "text": text,
            "format": "wav",
            "references": references,
            "reference_id": self.reference_id,
            "temperature": 0.7,
            "max_new_tokens": max_tokens,
            "streaming": False,
            "normalize": True,
            "chunk_length": 200,
        }
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/tts",
            params={"format": "msgpack"},
            data=ormsgpack.packb(payload),
            headers={"content-type": "application/msgpack"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        audio = response.content
        if not audio.startswith(b"RIFF"):
            raise RuntimeError(f"Voice server returned non-audio data ({len(audio)} bytes)")
        elapsed = time.perf_counter() - started
        print(f"[voice] synthesized {len(text)} chars in {elapsed:.1f}s")
        return audio
