"""Maya TTS: a macOS desktop app for Maya 2 Native (12 Indic languages + English), running on MLX.

    .venv/bin/python -m app.main
"""
from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QCheckBox, QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import player as P  # noqa: E402

LANGUAGES = [("ta", "Tamil"), ("hi", "Hindi"), ("en", "English"), ("te", "Telugu"), ("kn", "Kannada"),
             ("ml", "Malayalam"), ("bn", "Bengali"), ("mr", "Marathi"), ("gu", "Gujarati"), ("pa", "Punjabi"),
             ("or", "Odia"), ("as", "Assamese")]
SAMPLES = {
    "ta": "வணக்கம்! இன்று உங்கள் நாள் எப்படி இருக்கிறது? நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    "hi": "नमस्ते! आज आपका दिन कैसा है? मैं आपकी क्या मदद कर सकता हूँ?",
    "en": "Hello! How is your day going? How can I help you today?",
    "te": "నమస్కారం! ఈ రోజు మీ రోజు ఎలా ఉంది?",
    "kn": "ನಮಸ್ಕಾರ! ಇಂದು ನಿಮ್ಮ ದಿನ ಹೇಗಿದೆ?",
    "ml": "നമസ്കാരം! ഇന്ന് നിങ്ങളുടെ ദിവസം എങ്ങനെയുണ്ട്?",
    "bn": "নমস্কার! আজ আপনার দিন কেমন কাটছে?",
    "mr": "नमस्कार! आज तुमचा दिवस कसा आहे?",
    "gu": "નમસ્તે! આજે તમારો દિવસ કેવો છે?",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਅੱਜ ਤੁਹਾਡਾ ਦਿਨ ਕਿਵੇਂ ਹੈ?",
    "or": "ନମସ୍କାର! ଆଜି ଆପଣଙ୍କ ଦିନ କେମିତି ଅଛି?",
    "as": "নমস্কাৰ! আজি আপোনাৰ দিনটো কেনে?",
}


class Worker(QObject):
    """Owns the model; every MLX call happens on this one thread."""
    loaded = Signal(list)                 # [(name, meta)]
    status = Signal(str)
    audio = Signal(int, object)           # job id, int16 chunk
    done = Signal(dict)
    failed = Signal(str)
    voice_added = Signal(str, dict)

    def __init__(self):
        super().__init__()
        self.jobs = queue.Queue()
        self.cancel = threading.Event()
        self.tts = None
        threading.Thread(target=self._run, daemon=True, name="maya-mlx").start()

    def _run(self):
        try:
            self.status.emit("Loading Maya 2 Native (first run downloads ~5.4 GB)…")
            from maya2_mlx.model import Maya2TTS
            self.tts = Maya2TTS.from_pretrained(log=lambda m: self.status.emit(m))
            self.tts.generate("Ready.", lang="en", voice="Ahana")            # warm-up: compiles the Metal kernels
            self.loaded.emit([(n, v.meta) for n, v in self.tts.voices.items()])
            self.status.emit(f"Ready · loaded in {self.tts.load_s:.0f} s")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Could not load the model: {type(e).__name__}: {e}")
            return
        while True:
            kind, args = self.jobs.get()
            try:
                if kind == "speak":
                    self._speak(*args)
                elif kind == "clone":
                    self._clone(*args)
            except Exception as e:  # noqa: BLE001
                self.failed.emit(f"{type(e).__name__}: {e}")

    def _speak(self, job, text, lang, voice, normalize, seed):
        info = {"job": job}
        self.cancel.clear()
        self.status.emit("Generating…")
        for pcm in self.tts.stream(text, lang, voice, normalize=normalize, seed=seed, cancel=self.cancel, info=info):
            self.audio.emit(job, pcm)
        self.done.emit(info)

    def _clone(self, name, path, transcript, lang):
        from maya2_mlx.voices import clone_voice
        self.status.emit(f"Encoding reference clip for “{name}”…")
        v = clone_voice(self.tts, name, path, transcript, lang)
        self.voice_added.emit(v.name, v.meta)
        self.status.emit(f"Added cloned voice “{name}”")


class CloneDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clone a voice")
        self.path = QLineEdit()
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path)
        row.addWidget(browse)
        self.name = QLineEdit()
        self.lang = QComboBox()
        for code, label in LANGUAGES:
            self.lang.addItem(label, code)
        self.transcript = QPlainTextEdit()
        self.transcript.setPlaceholderText("The exact words spoken in the clip")
        self.transcript.setFixedHeight(80)
        form = QFormLayout()
        form.addRow("Reference clip (3–10 s)", row)
        form.addRow("Transcript", self.transcript)
        form.addRow("Clip language", self.lang)
        form.addRow("Voice name", self.name)
        note = QLabel("Use a clean, single-speaker recording, and only clone a voice with the speaker's permission.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addWidget(buttons)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Reference clip", str(Path.home()), "Audio (*.wav *.flac *.mp3 *.m4a *.ogg)")
        if p:
            self.path.setText(p)
            if not self.name.text():
                self.name.setText(Path(p).stem)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Maya TTS")
        self.resize(760, 520)
        self.settings = QSettings("local", "Maya TTS")
        self.player = P.StreamPlayer()
        self.player.device = self.settings.value("output_device", None) or None
        self.player.monitor = self.settings.value("monitor", False, type=bool)
        self.player.on_done = lambda: self.worker.status.emit(self._final_status or "Ready")
        self.last_audio = []
        self._final_status = None
        self.job = 0
        self.worker = Worker()
        self.worker.loaded.connect(self._on_loaded)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.audio.connect(self._on_audio)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.voice_added.connect(self._on_voice_added)

        self.text = QPlainTextEdit()
        font = QFont()                      # system font: macOS falls back per script (Tamil Sangam MN, Kohinoor, ...)
        font.setPointSize(16)
        self.text.setFont(font)
        self.text.setPlaceholderText("Type or paste text…")
        self.lang = QComboBox()
        for code, label in LANGUAGES:
            self.lang.addItem(label, code)
        self.lang.currentIndexChanged.connect(self._on_lang)
        self.voice = QComboBox()
        self.voice.setMinimumWidth(260)
        self.voice.addItem("Loading…")
        self.normalize = QCheckBox("Normalise numbers, dates, currency")
        self.normalize.setChecked(True)
        self.seed = QSpinBox()
        self.seed.setRange(-1, 2 ** 31 - 1)
        self.seed.setValue(-1)
        self.seed.setSpecialValueText("random")
        self.speak = QPushButton("▶  Speak")
        self.speak.setDefault(True)
        self.speak.setEnabled(False)
        self.speak.clicked.connect(self._on_speak)
        self.stop = QPushButton("■  Stop")
        self.stop.clicked.connect(self._on_stop)
        self.save = QPushButton("Save WAV…")
        self.save.setEnabled(False)
        self.save.clicked.connect(self._on_save)
        self.clone = QPushButton("Clone voice…")
        self.clone.setEnabled(False)
        self.clone.clicked.connect(self._on_clone)

        top = QHBoxLayout()
        top.addWidget(QLabel("Language"))
        top.addWidget(self.lang)
        top.addSpacing(12)
        top.addWidget(QLabel("Voice"))
        top.addWidget(self.voice, 1)
        top.addWidget(self.clone)
        self.output = QComboBox()
        self.output.setMinimumWidth(260)
        self.output.setToolTip("Where the speech plays. Pick a virtual or loopback device (for example a meeting "
                               "app's audio device) to make apps that listen to it as a microphone hear the speech.")
        self.output.activated.connect(self._on_output)
        rescan = QPushButton("↻")
        rescan.setFixedWidth(36)
        rescan.setToolTip("Rescan audio devices")
        rescan.clicked.connect(self._on_rescan)
        self.monitor = QCheckBox()
        self.monitor.setChecked(self.player.monitor)
        self.monitor.toggled.connect(self._on_monitor)
        out = QHBoxLayout()
        out.addWidget(QLabel("Output"))
        out.addWidget(self.output, 1)
        out.addWidget(rescan)
        out.addWidget(self.monitor)
        opts = QHBoxLayout()
        opts.addWidget(self.normalize)
        opts.addStretch(1)
        opts.addWidget(QLabel("Seed"))
        opts.addWidget(self.seed)
        buttons = QHBoxLayout()
        buttons.addWidget(self.speak)
        buttons.addWidget(self.stop)
        buttons.addStretch(1)
        buttons.addWidget(self.save)
        lay = QVBoxLayout()
        lay.addLayout(top)
        lay.addWidget(self.text, 1)
        lay.addLayout(opts)
        lay.addLayout(out)
        lay.addLayout(buttons)
        w = QWidget()
        w.setLayout(lay)
        self.setCentralWidget(w)
        self._fill_outputs()
        self._on_lang()

        act = QAction("Speak", self)
        act.setShortcut(QKeySequence("Ctrl+Return"))      # ⌘↩ on macOS
        act.triggered.connect(lambda: self.speak.isEnabled() and self._on_speak())
        self.addAction(act)
        esc = QAction("Stop", self)
        esc.setShortcut(QKeySequence(Qt.Key_Escape))
        esc.triggered.connect(self._on_stop)
        self.addAction(esc)

    # ---------------------------------------------------------------- helpers
    def _fill_outputs(self):
        """Rebuild the Output list; a remembered device that is unplugged stays listed (marked) and falls back to the
        system default until it returns."""
        default = P.default_output_name()
        names = [n for _, n in P.output_devices()]
        self.output.clear()
        self.output.addItem(f"System default ({default})", None)
        for n in names:
            self.output.addItem(n, n)
        chosen = self.player.device
        if chosen and chosen not in names:
            self.output.addItem(f"{chosen} (not connected — using system default)", chosen)
        self.output.setCurrentIndex(max(0, self.output.findData(chosen)) if chosen else 0)
        self.monitor.setText(f"Also play on {default}")
        self.monitor.setEnabled(bool(chosen) and chosen != default)

    def _on_output(self):
        self.player.device = self.output.currentData()
        self.settings.setValue("output_device", self.player.device or "")
        self._fill_outputs()

    def _on_monitor(self, on):
        self.player.monitor = on
        self.settings.setValue("monitor", on)

    def _on_rescan(self):
        if self.player.active:
            self.statusBar().showMessage("Stop playback before rescanning devices")
            return
        P.rescan_devices()
        self._fill_outputs()
        self.statusBar().showMessage("Audio devices rescanned")

    def _on_lang(self):
        code = self.lang.currentData()
        cur = self.text.toPlainText().strip()
        if not cur or cur in SAMPLES.values():
            self.text.setPlainText(SAMPLES.get(code, ""))
        self._mark_voices()

    def _mark_voices(self):
        """Grey-label voices whose training data did not include the language (they still work, cross-lingually)."""
        code = self.lang.currentData()
        for i in range(self.voice.count()):
            meta = self.voice.itemData(i)
            if not isinstance(meta, dict):
                continue
            label = meta.get("_label")
            langs = meta.get("langs")
            native = not langs or code in langs or meta.get("cloned")
            self.voice.setItemText(i, label if native else f"{label}  (cross-lingual)")

    def _add_voice_item(self, name, meta):
        desc = meta.get("description", "")
        label = f"{name} — {desc}" if desc else name
        self.voice.addItem(label, {**meta, "_label": label, "_name": name})

    # ---------------------------------------------------------------- slots
    def _on_loaded(self, voices):
        self.voice.clear()
        for name, meta in voices:
            self._add_voice_item(name, meta)
        self._mark_voices()
        self.speak.setEnabled(True)
        self.clone.setEnabled(True)

    def _on_voice_added(self, name, meta):
        self._add_voice_item(name, meta)
        self.voice.setCurrentIndex(self.voice.count() - 1)
        self._mark_voices()

    def _on_speak(self):
        text = self.text.toPlainText().strip()
        if not text:
            return
        self.worker.cancel.set()                 # interrupt anything still generating
        try:
            self.player.start()
        except Exception as e:  # noqa: BLE001  (device unplugged, busy, ...)
            self._on_failed(f"Could not open the output device: {e}")
            return
        self.statusBar().showMessage("Playing on " + " + ".join(self.player.targets()))
        self.last_audio = []
        self._final_status = None
        self.job += 1
        meta = self.voice.currentData() or {}
        seed = self.seed.value()
        self.worker.jobs.put(("speak", (self.job, text, self.lang.currentData(), meta.get("_name", "Ahana"),
                                        self.normalize.isChecked(), None if seed < 0 else seed)))

    def _on_audio(self, job, pcm):
        if job != self.job:                      # a cancelled request's last chunks
            return
        self.player.push(pcm)
        self.last_audio.append(pcm)

    def _on_done(self, info):
        if info.get("job") != self.job:
            return
        self.player.finish()
        self.save.setEnabled(bool(self.last_audio))
        if info.get("audio_s"):
            reasons = ", ".join(s["reason"] for s in info.get("stops", []))
            self._final_status = (f"{info['audio_s']:.1f} s of audio in {info['wall_s']:.1f} s "
                                  f"(RTF {info['rtf']:.2f}, first audio {info['ttfa_s']:.2f} s) · stop: {reasons} · "
                                  f"on {' + '.join(self.player.targets())}")
            self.statusBar().showMessage(self._final_status)

    def _on_stop(self):
        self.worker.cancel.set()
        self.player.stop()
        self.statusBar().showMessage("Stopped")

    def _on_failed(self, msg):
        self.player.finish()
        self.statusBar().showMessage(msg)
        QMessageBox.warning(self, "Maya TTS", msg)

    def _on_save(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save audio", str(Path.home() / "maya_tts.wav"), "WAV (*.wav)")
        if p:
            import soundfile as sf
            sf.write(p, np.concatenate(self.last_audio), 24000, subtype="PCM_16")
            self.statusBar().showMessage(f"Saved {p}")

    def _on_clone(self):
        d = CloneDialog(self)
        d.lang.setCurrentIndex(self.lang.currentIndex())
        if d.exec() != QDialog.Accepted:
            return
        path, text, name = d.path.text().strip(), d.transcript.toPlainText().strip(), d.name.text().strip()
        if not (path and text and name):
            QMessageBox.warning(self, "Maya TTS", "A clip, its transcript and a name are all needed.")
            return
        self.worker.jobs.put(("clone", (name, path, text, d.lang.currentData())))

    def closeEvent(self, e):
        self.worker.cancel.set()
        self.player.stop()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Maya TTS")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
