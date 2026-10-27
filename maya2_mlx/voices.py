"""Zero-shot voice cloning from a 3-10 s reference clip and its exact transcript.

The clip is Mimi-encoded on MLX and packed as an in-context reference span ("icl"). The release's default clone mode,
"icl_spk", also feeds an ECAPA-192 speaker embedding to the speaker anchor and the depth memory; that encoder
(speechbrain, PyTorch) is used only when it is installed, otherwise the clone falls back to "icl".
"""
from __future__ import annotations

from math import gcd

import numpy as np

from .prompt import Voice

SR = 24000


def load_audio(path):
    import soundfile as sf
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        w = resample_poly(w, SR // g, int(sr) // g).astype(np.float32)
    return w


def ecapa_embedding(w24):
    """ECAPA-192 (speechbrain/spkrec-ecapa-voxceleb, the release's recipe) or None when speechbrain is unavailable."""
    try:
        import torch
        from huggingface_hub import snapshot_download
        from maya3.api import Ecapa
    except ImportError:
        return None
    d = snapshot_download("speechbrain/spkrec-ecapa-voxceleb", allow_patterns=["embedding_model.ckpt", "hyperparams.yaml"])
    with torch.inference_mode():
        return Ecapa(d, "cpu")(w24)


def clone_voice(tts, name, path, transcript, lang, use_ecapa=True):
    w = load_audio(path)
    if len(w) < SR:
        raise ValueError("the reference clip is shorter than 1 s; 3-10 s works best")
    if len(w) > 30 * SR:
        raise ValueError("the reference clip is longer than 30 s; 3-10 s works best")
    peak = float(np.abs(w).max())
    if peak > 0.99:
        w = w * (0.99 / peak)
    codes = tts.encode_reference(w)
    ecapa = ecapa_embedding(w) if use_ecapa else None
    mode = "icl_spk" if ecapa is not None else "icl"
    v = Voice(name, mode, meta={"name": name, "cloned": True, "description": f"cloned ({mode}), {lang} reference"},
              ref_text=transcript.strip(), ref_lang=lang, ref_codes=codes, ecapa=ecapa)
    tts.add_voice(v)
    return v
