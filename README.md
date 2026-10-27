# Maya TTS for Mac

A macOS desktop app that speaks 11 Indic languages and English with the open
[Maya 2 Native](https://huggingface.co/maya-research/maya2-native) model (Maya Research), entirely on your Mac. Speech
starts playing within about half a second of pressing Speak, on any output device you choose, including a virtual
microphone that call and recording apps can listen to.

| | |
|---|---|
| Languages | Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu |
| Voices | 20 built-in voices, plus zero-shot cloning from a 3–10 s clip |
| Speed (M1 Max) | real-time factor 0.65 (about 1.5x faster than real time), first audio generated in about 0.2 s |
| Memory | about 5.6 GB peak |

## What it's built with

The app is Python. The window is [Qt for Python (PySide6)](https://doc.qt.io/qtforpython-6/), the model runs on
Apple's [MLX](https://github.com/ml-explore/mlx) through [mlx-audio](https://github.com/Blaizzy/mlx-audio), and audio
goes out through [sounddevice](https://python-sounddevice.readthedocs.io/) (Core Audio). There is no Xcode project and
no Swift code; the optional `.app` bundle is a small launcher script that starts the Python app.

Python keeps the app next to the model code it needs: the release's text normaliser and stop guards are Python, and
mlx-audio already provides the Qwen3 backbone and the Mimi codec. A native SwiftUI front end is possible, but it would
either drive a Python process or need the model ported again to `mlx-swift`.

## Build from source

You need an Apple Silicon Mac on macOS 14 or later, about 7 GB of free disk space, and
[uv](https://docs.astral.sh/uv/).

1. Install uv (or `brew install uv`):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Get the code and create the environment:

   ```bash
   git clone https://github.com/ashwanthkumar/maya-tts-mac.git
   cd maya-tts-mac
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

   This installs MLX, mlx-audio, PySide6, sounddevice and a few small libraries. PyTorch is not needed.

3. Run it:

   ```bash
   .venv/bin/python -m app.main
   ```

   The first launch downloads the model (about 5.4 GB) and the Mimi codec into your Hugging Face cache
   (`~/.cache/huggingface`). Later launches load in a few seconds.

4. Optional: build a double-clickable app for your own Mac.

   ```bash
   scripts/make_app.sh
   open "dist/Maya TTS.app"
   ```

   This creates `dist/Maya TTS.app` with its own icon. The bundle runs this folder's `.venv`, so it works only on the
   Mac that built it and while the folder stays where it is; rebuild it after moving the folder. Build it yourself
   rather than copying someone else's bundle. It is not checked into the repository.

## Using it

- **Language and voice.** Pick one of 12 languages and 20 voices. Voices marked *cross-lingual* weren't trained on that
  language but still speak it.
- **⌘↩** speaks and **Esc** stops. Playback starts while the rest is still being generated.
- **Output** chooses where the speech plays: the system default (whatever it is when you press Speak) or a specific
  device. To feed the speech into a call or a recorder as a microphone, pick a virtual or loopback device, then select
  the same device as the microphone in the other app. Meeting apps often install one (for example Zoom's or Teams'
  audio device), and `brew install blackhole-2ch` adds a general one. **Also play on <default device>** sends a copy to
  your speakers or headphones so you hear it too. Mono speech goes to both channels of a stereo device. **↻** rescans
  devices after you plug something in. The choice is remembered; if the device is unplugged, the app falls back to
  the system default.
- **Normalise** reads numbers, currency, dates and times as words, using the release's own normaliser.
- **Seed** reproduces an output exactly; leave it on *random* otherwise.
- **Save WAV…** saves the last output (24 kHz, 16-bit mono).
- **Clone voice…** takes a clean 3–10 s single-speaker clip, its exact transcript and its language; the clone can speak
  any of the 12 languages. Only clone a voice with the speaker's permission. Without PyTorch, a clone uses the
  reference clip in context only (`icl`). If `speechbrain` and PyTorch are installed in `.venv`, it also uses the
  release's ECAPA speaker embedding (`icl_spk`, the release's default), which gives closer similarity.
- **Logs.** The `.app` writes `~/Library/Logs/maya-tts.log` itself and keeps it under 50 MB the whole time: every
  write is counted, and when the log passes the cap it is cut to its last 1 MB (`app/logcap.py`). Run from a terminal,
  the app prints to the terminal instead.

Generation is sampled, so the same text varies slightly between runs (see [Quality](#quality)).

## How it works

Maya 2 Native produces speech as a stream of Mimi codec frames, 12.5 per second (80 ms each), each made of 32 codes.
A Qwen3-1.7B backbone predicts the first code of each frame, a smaller depth decoder fills in the other 31 conditioned
on the voice, Mimi turns the 32 codes into 1,920 samples at 24 kHz, and the frame is fed back into the backbone as
the input for the next one.

```
            ┌──────────────────────── frame fed back ────────────────────────┐
            ▼                                                                │
  text ─▶ Backbone ─▶ first code + stop check ─▶ Depth decoder ─▶ Mimi decode ─▶ playback
          (GPU)       (the CPU waits here)        codes 1–31 (GPU)   (GPU)
```

The official engine targets NVIDIA GPUs (CUDA graphs, Triton kernels). This port runs the same loop on the Mac's GPU,
and the CPU waits only once per frame, for the stop decision on the first code. The depth decoder's 31 levels, the
next backbone step and the frame's Mimi decode are queued lazily and run while the CPU checks the guards. Because the
voice enters the depth decoder as a single memory token, its cross-attention reduces to a per-voice constant that is
computed once per voice.

```
maya2_mlx/           the model in MLX, built on mlx-audio
  prompt.py          prompt packing (numpy port of the release's packer), text splitting, named voices
  model.py           backbone + first-code head (mlx-audio's Qwen3), audio-frame embeddings, speaker anchor,
                     depth decoder, the decode loop with the official sampler, stop rules and loop guards,
                     streaming Mimi decode (mlx-audio)
  voices.py          voice cloning
app/                 the Qt window (main.py), the streaming multi-device player (player.py), the capped log
                     (logcap.py) and the .app entry point (launch.py)
scripts/             make_app.sh builds the .app bundle; make_icon.py draws its icon
third_party/maya2-native/   the official inference code (Apache-2.0), vendored at the commit in
                     third_party/MAYA2_NATIVE_COMMIT; the app imports only its text normaliser and guards (pure
                     Python); the rest is the test reference
tests/               checks against the official PyTorch implementation
```

## The bug that sounded almost right

The first working port produced speech that sounded natural but garbled about one word in four, while every
numerical check against the official code passed. The cause was one constant: the epsilon inside the RMSNorm that
scales each audio frame's embedding before it re-enters the backbone (`audio_in`, a `torch.nn.RMSNorm(eps=None)`).

With `eps=None`, PyTorch derives the epsilon from the data type. The model runs in bfloat16, so the obvious choice is
bfloat16's epsilon, 2^-7. But PyTorch computes this op in fp32 and uses the fp32 epsilon, 1.2e-7. These embeddings are
tiny (pre-norm mean square about 2.5e-3), so 2^-7 shrank every frame's input about 2x. Whisper character error rate,
5 takes per sentence:

| normalisation epsilon | English | Hindi | Tamil |
|---|---|---|---|
| 2^-7 (bfloat16) | 0.198 | 0.436 | 0.516 |
| 1.2e-7 (what PyTorch uses) | 0.000 | 0.041 | 0.009 |

## Verification

The official code needs PyTorch and `transformers==5.6.2`, while mlx-audio needs a newer transformers, so the
reference checks run in a separate environment:

```bash
uv venv --python 3.12 .venv-ref
uv pip install --python .venv-ref/bin/python torch==2.8.0 torchaudio==2.8.0 transformers==5.6.2 "tokenizers<0.23" \
    scipy soundfile speechbrain==1.1.2 huggingface_hub safetensors pyyaml mlx==0.32.4
uv pip install --python .venv-ref/bin/python --no-deps mlx-audio==0.6.2

.venv-ref/bin/python tests/test_parity.py      # MLX vs PyTorch in fp32
.venv-ref/bin/python tests/quality_sweep.py 5  # decoding settings scored by Whisper
.venv-ref/bin/python tests/ref_decode_check.py "<text>" <lang> <Voice> <whisper language> [seeds...]
```

`test_parity.py` checks, for a named voice and for same-language and cross-language clones: prompt tokens identical;
prompt embeddings within 5e-7; backbone hidden state and first-code logits within 1e-4; greedy depth codes 32/32. Over
30 teacher-forced frames of real speech the logits stay within 1e-4, and mlx-audio's Mimi decodes the same codes as
`transformers`' `kyutai/mimi` at 53 dB SNR.

## Quality

Measured with `tests/quality_sweep.py`: 5 seeds per sentence, Whisper large-v3-turbo as the judge, mean character
error rate (CER) against the input text, M1 Max, bf16 weights:

| decoding | English | Hindi | Tamil |
|---|---|---|---|
| official engine sampler (shipped) | 0.000 | 0.041 | 0.009 |
| depth levels >= 9 greedy | 0.000 | 0.036 | 0.005 |
| fully greedy depth | 0.000 | 0.031 | 0.014 |

The leftover Hindi errors are mostly spelling variants in the transcript, such as ओर्डर for ऑर्डर. All settings are
within noise of each other, so the app keeps the official engine's sampler. An occasional take can still slip; press
Speak again.

## Credits

Maya 2 Native weights and inference code: Apache-2.0, [Maya Research](https://huggingface.co/maya-research/maya2-native)
(vendored copy and its license in `third_party/maya2-native`). Mimi codec: CC-BY-4.0,
[Kyutai](https://huggingface.co/kyutai/mimi). Built on [mlx-audio](https://github.com/Blaizzy/mlx-audio) and
[MLX](https://github.com/ml-explore/mlx).
