"""Streaming PCM playback on one or more output devices."""
from __future__ import annotations

import collections
import threading

import numpy as np
import sounddevice as sd


def output_devices():
    """[(index, name)] of every device with output channels, in PortAudio order."""
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]


def default_output_name():
    try:
        return sd.query_devices(kind="output")["name"]
    except Exception:
        return "default output"


def rescan_devices():
    """Re-enumerate audio devices (PortAudio caches the list at start-up). Only call with no stream open."""
    sd._terminate()
    sd._initialize()


def resolve(name):
    """Device index for a device name, or None (= system default) when it is gone."""
    if name is None:
        return None
    for i, n in output_devices():
        if n == name:
            return i
    return None


class _Sink:
    """One output stream with its own queue, so a slow device never starves another."""

    def __init__(self, device, sr, prebuffer, on_done):
        self.prebuffer = prebuffer
        self.chunks = collections.deque()
        self.head = 0
        self.queued = 0
        self.started = False
        self.finished = False
        self.lock = threading.Lock()
        self.on_done = on_done
        info = sd.query_devices(device, kind="output") if device is None else sd.query_devices(device)
        self.channels = min(2, info["max_output_channels"])      # mono is duplicated onto L+R of a stereo device
        self.name = info["name"]
        self.stream = sd.OutputStream(samplerate=sr, channels=self.channels, dtype="int16", device=device,
                                      callback=self._callback, latency="high", finished_callback=on_done)
        self.stream.start()

    def push(self, pcm):
        with self.lock:
            self.chunks.append(pcm)
            self.queued += len(pcm)

    def finish(self):
        with self.lock:
            self.finished = True

    def close(self):
        self.stream.abort()
        self.stream.close()

    def _callback(self, out, frames, time_info, status):
        out.fill(0)
        with self.lock:
            if not self.started:
                if self.queued < self.prebuffer and not self.finished:
                    return
                self.started = True
            filled = 0
            while filled < frames and self.chunks:
                c = self.chunks[0]
                take = min(frames - filled, len(c) - self.head)
                out[filled:filled + take] = c[self.head:self.head + take, None]
                filled += take
                self.head += take
                self.queued -= take
                if self.head == len(c):
                    self.chunks.popleft()
                    self.head = 0
            if self.finished and not self.chunks:
                raise sd.CallbackStop


class StreamPlayer:
    """Feed int16 mono chunks with push(); they play as they arrive on the chosen output device, and optionally also
    on the system default output (to hear what is being sent to a virtual microphone).

    Streams are opened per utterance, so device changes (headphones, a new default) apply on the next Speak."""

    def __init__(self, sample_rate=24000, prebuffer_s=0.4):
        self.sr = sample_rate
        self.prebuffer = int(prebuffer_s * sample_rate)
        self.device = None                 # output device name; None = system default
        self.monitor = False               # also play on the system default output
        self._sinks = []
        self._pending = 0
        self._gen = 0                      # start() generation: finished callbacks of aborted streams are ignored
        self._lock = threading.Lock()
        self.on_done = None                # called (from an audio thread) when every sink has played everything

    @property
    def active(self):
        return bool(self._sinks)

    def targets(self):
        """Names the next start() will play on."""
        dev = resolve(self.device)
        names = [default_output_name() if dev is None else self.device]
        if self.monitor and dev is not None and self.device != default_output_name():
            names.append(default_output_name())
        return names

    def start(self):
        self.stop()
        dev = resolve(self.device)
        devices = [dev]
        if self.monitor and dev is not None and self.device != default_output_name():
            devices.append(None)
        with self._lock:
            self._gen += 1
            self._pending = len(devices)
            gen = self._gen
        self._sinks = [_Sink(d, self.sr, self.prebuffer, lambda g=gen: self._sink_done(g)) for d in devices]

    def push(self, pcm: np.ndarray):
        pcm = np.asarray(pcm, np.int16)
        for s in self._sinks:
            s.push(pcm)

    def finish(self):
        """No more audio is coming: start playing whatever is buffered and stop when drained."""
        for s in self._sinks:
            s.finish()

    def stop(self):
        with self._lock:
            self._gen += 1
        sinks, self._sinks = self._sinks, []
        for s in sinks:
            s.close()

    def _sink_done(self, gen):
        with self._lock:
            if gen != self._gen:
                return
            self._pending -= 1
            last = self._pending == 0
        if last and self.on_done is not None:
            self.on_done()
