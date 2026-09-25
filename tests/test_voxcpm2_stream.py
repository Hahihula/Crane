#!/usr/bin/env python3
"""Call crane-serve VoxCPM2 streaming TTS and save a playable WAV file."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream VoxCPM2 PCM from crane-serve into a local WAV file."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080/v1/audio/speech",
        help="crane-serve speech endpoint",
    )
    parser.add_argument(
        "--voice",
        default="voice_preview_adam",
        help="Built-in voice name without its file extension",
    )
    parser.add_argument(
        "--text",
        default="你好，这是固定音色的流式语音测试。",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("voxcpm2-stream.wav"),
        help="Destination WAV path",
    )
    parser.add_argument("--cfm-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Maximum bytes read per receive operation",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play PCM through sounddevice while it is being received",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="sounddevice output device name or index (default: system output)",
    )
    parser.add_argument(
        "--play-buffer-ms",
        type=int,
        default=500,
        help="Audio buffered before playback starts (default: 500 ms)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List sounddevice devices and exit",
    )
    return parser.parse_args()


class SoundDevicePlayer:
    """Non-blocking PCM16 playback with a small jitter buffer."""

    def __init__(
        self, sample_rate: int, device: str | None = None, buffer_ms: int = 500
    ) -> None:
        try:
            import sounddevice
        except ImportError as error:
            raise RuntimeError(
                "--play requires sounddevice; install it with: "
                "python3 -m pip install sounddevice"
            ) from error
        resolved_device: str | int | None = device
        if device is not None and device.isdecimal():
            resolved_device = int(device)
        self.sample_rate = sample_rate
        self._bytes_per_second = sample_rate * 2
        self._prebuffer_bytes = max(0, buffer_ms) * self._bytes_per_second // 1000
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._playback_started = self._prebuffer_bytes == 0
        self.underflows = 0

        def callback(outdata, frames, _time_info, status) -> None:
            requested = frames * 2
            with self._lock:
                if not self._playback_started and len(self._buffer) >= self._prebuffer_bytes:
                    self._playback_started = True
                available = min(requested, len(self._buffer)) if self._playback_started else 0
                if available:
                    outdata[:available] = self._buffer[:available]
                    del self._buffer[:available]
                if available < requested:
                    outdata[available:requested] = bytes(requested - available)
                    if self._playback_started:
                        self.underflows += 1

        self._stream = sounddevice.RawOutputStream(
            device=resolved_device,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            latency="high",
            blocksize=2048,
            callback=callback,
        )
        self._stream.start()

    def write(self, pcm: bytes) -> None:
        if len(pcm) % 2:
            raise RuntimeError("received an odd number of PCM16 bytes")
        with self._lock:
            self._buffer.extend(pcm)

    @property
    def buffered_ms(self) -> float:
        with self._lock:
            return len(self._buffer) / self._bytes_per_second * 1000

    def close(self) -> None:
        with self._lock:
            if self._buffer:
                self._playback_started = True
        deadline = time.monotonic() + self.buffered_ms / 1000 + 1.0
        while self.buffered_ms > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self._stream.stop()
        self._stream.close()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        try:
            import sounddevice
        except ImportError as error:
            print(f"sounddevice is unavailable: {error}", file=sys.stderr)
            return 1
        print(sounddevice.query_devices())
        return 0
    payload = {
        "model": "voxcpm2",
        "input": args.text,
        "voice": args.voice,
        "response_format": "pcm",
        "stream": True,
        "cfm_steps": args.cfm_steps,
        "cfg_scale": args.cfg_scale,
        "max_tokens": args.max_tokens,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never leave a previous, partial WAV looking like the result of this run.
    args.output.unlink(missing_ok=True)
    total_bytes = 0
    chunk_count = 0
    request_started = time.perf_counter()
    first_chunk_at: float | None = None
    previous_chunk_at: float | None = None
    player: SoundDevicePlayer | None = None
    if args.play:
        try:
            # Start CoreAudio before the potentially long model warm-up. On macOS,
            # opening AUHAL after waiting on the HTTP response can fail with -9986.
            player = SoundDevicePlayer(48_000, args.device, args.play_buffer_ms)
        except Exception as error:
            print(
                f"warning: live playback unavailable ({error}); "
                "continuing to save the complete WAV",
                file=sys.stderr,
            )
            player = None
    try:
        with urllib.request.urlopen(request) as response:
            headers_at = time.perf_counter()
            content_type = response.headers.get_content_type()
            if content_type != "audio/pcm":
                body = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"expected audio/pcm, received {content_type}: {body}"
                )
            sample_rate = int(response.headers.get("X-Sample-Rate", "48000"))
            if player is not None and player.sample_rate != sample_rate:
                player.close()
                player = None
                print(
                    f"warning: server returned {sample_rate} Hz but playback was "
                    "opened at 48000 Hz; saving WAV without live playback",
                    file=sys.stderr,
                )
            read_chunk = getattr(response, "read1", response.read)
            print(
                f"HTTP ready in {(headers_at - request_started) * 1000:.1f} ms; "
                f"PCM16 mono {sample_rate} Hz"
            )
            with wave.open(str(args.output), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # signed PCM16 little-endian
                wav.setframerate(sample_rate)
                carry = b""
                while network_chunk := read_chunk(args.chunk_size):
                    received_at = time.perf_counter()
                    chunk = carry + network_chunk
                    if len(chunk) % 2:
                        carry = chunk[-1:]
                        chunk = chunk[:-1]
                    else:
                        carry = b""
                    if not chunk:
                        continue
                    if first_chunk_at is None:
                        first_chunk_at = received_at
                    interval_ms = (
                        0.0
                        if previous_chunk_at is None
                        else (received_at - previous_chunk_at) * 1000
                    )
                    previous_chunk_at = received_at
                    chunk_count += 1
                    wav.writeframesraw(chunk)
                    if player is not None:
                        try:
                            player.write(chunk)
                        except Exception as error:
                            print(
                                f"warning: live playback stopped ({error}); "
                                "continuing to save the complete WAV",
                                file=sys.stderr,
                            )
                            try:
                                player.close()
                            except Exception:
                                pass
                            player = None
                    total_bytes += len(chunk)
                    chunk_frames = len(chunk) // 2
                    chunk_audio_ms = chunk_frames / sample_rate * 1000
                    audio_seconds = total_bytes / (sample_rate * 2)
                    wall_seconds = received_at - request_started
                    realtime = audio_seconds / max(wall_seconds, 1e-9)
                    print(
                        f"chunk={chunk_count:04d} frames={chunk_frames:6d} "
                        f"audio={chunk_audio_ms:7.1f}ms interval={interval_ms:7.1f}ms "
                        f"total={audio_seconds:7.2f}s realtime={realtime:5.2f}x"
                        + (
                            f" buffer={player.buffered_ms:7.1f}ms "
                            f"underflows={player.underflows}"
                            if player is not None
                            else ""
                        ),
                        flush=True,
                    )
                if carry:
                    raise RuntimeError(
                        "stream ended with an incomplete PCM16 sample (1 trailing byte)"
                    )
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"HTTP {error.code}: {body}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"request failed: {error}", file=sys.stderr)
        return 1
    finally:
        if player is not None:
            try:
                player.close()
            except Exception as error:
                print(f"warning: playback close failed: {error}", file=sys.stderr)

    duration = total_bytes / (sample_rate * 2)
    ttfa_ms = (
        (first_chunk_at - request_started) * 1000 if first_chunk_at is not None else 0.0
    )
    elapsed = time.perf_counter() - request_started
    print(
        f"saved {args.output.resolve()} ({duration:.2f}s, {sample_rate} Hz mono PCM16)\n"
        f"chunks={chunk_count}, TTFA={ttfa_ms:.1f}ms, elapsed={elapsed:.2f}s, "
        f"generation realtime={duration / max(elapsed, 1e-9):.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
