"""Prompt packing for Maya 2 Native, without torch.

A numpy port of the release's packer (tts/data/packing_v2.PackerV2._span + collate_v2 + truncate_prompt_batch, via
maya3.prompt.V5Builder / NamedSpeakerBuilder). One span is

    [<|lang:xx|> if first] text-tokens <|audio|> frames... <aeos>

and a prompt is the voice's reference span (cloned voices only) followed by the target span cut right after its
<|audio|> sentinel, the position whose hidden state generates frame 0. The speaker anchor is added at every span's
<|audio|> sentinel. tests/test_parity.py checks this against the original code token for token.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

N_Q = 32
N_CODES = 2048
AEOS_CLASS = N_CODES
KIND_TEXT, KIND_AUDIO, KIND_AEOS = 0, 1, 3
FPS = 12.5
LANGS = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
TERMINALS = "।.!?;।॥"
PRIOR_DEFAULT = {"a": 2.0, "b": 0.72, "resid_p99": 90.0}


@dataclass
class Voice:
    """A named (trained) voice, or a clone: reference clip codes (32, F) + transcript + language (+ ECAPA-192)."""
    name: str
    mode: str = "named"                       # named | icl | icl_spk
    meta: dict = field(default_factory=dict)
    ref_text: Optional[str] = None
    ref_lang: Optional[str] = None
    ref_codes: Optional[np.ndarray] = None
    ecapa: Optional[np.ndarray] = None        # (192,) L2-normalised, icl_spk only

    @property
    def uses_speaker(self):
        return self.mode == "icl_spk" and self.ecapa is not None


@dataclass
class Prompt:
    tokens: np.ndarray        # (T,) int64 text ids (0 at audio / aeos positions)
    kind: np.ndarray          # (T,) int8
    codes: np.ndarray         # (T, 32) int64, -1 at non-audio positions
    sentinels: np.ndarray     # positions of the <|audio|> sentinels (speaker-anchor targets)


class Packer:
    def __init__(self, backbone_dir):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(Path(backbone_dir) / "tokenizer.json"))
        self.audio_id = self._id("<|audio|>")
        self.lang_id = {l: self._id(f"<|lang:{l}|>") for l in LANGS}

    def _id(self, t):
        i = self.tok.token_to_id(t)
        if i is None:
            raise ValueError(f"tokenizer lacks {t}")
        return i

    def span(self, text, lang, codes, first):
        """PackerV2._span: codes (32, F) -> (tokens, kind, codes) with the trailing <aeos> position."""
        if not text.strip():
            raise ValueError("empty text")
        head = ([self.lang_id[lang]] if first else []) + self.tok.encode(text, add_special_tokens=False).ids
        head.append(self.audio_id)
        n, F = len(head), codes.shape[1]
        L = n + F + 1
        t = np.zeros(L, np.int64)
        t[:n] = head
        k = np.zeros(L, np.int8)
        k[n:n + F] = KIND_AUDIO
        k[-1] = KIND_AEOS
        c = np.full((L, N_Q), -1, np.int64)
        c[n:n + F] = codes.T
        c[-1, 0] = AEOS_CLASS
        return t, k, c

    def prompt(self, voice: Voice, text, lang):
        text = text.strip()
        if voice.mode == "named":
            text = f"{{{voice.meta.get('name', voice.name)}}}: {text}"
        spans = []
        if voice.mode in ("icl", "icl_spk"):
            spans.append(self.span(voice.ref_text, voice.ref_lang, np.asarray(voice.ref_codes, np.int64), first=True))
        # the target's span carries its own language tag when it is first, or when the reference is in another language
        first = not spans or voice.ref_lang != lang
        spans.append(self.span(text, lang, np.zeros((N_Q, 1), np.int64), first=first))
        t, k, c = (np.concatenate([s[j] for s in spans]) for j in range(3))
        cut = len(t) - 2                        # drop the target's dummy frame and its <aeos>
        t, k, c = t[:cut], k[:cut], c[:cut]
        sentinels = np.nonzero((k == KIND_TEXT) & (t == self.audio_id))[0]
        return Prompt(t, k, c, sentinels)


# ---------------------------------------------------------------- length prior + text splitting (scripts/s2_serve_controls)
def load_prior(path):
    prior = {}
    if path and Path(path).exists():
        for lang, d in json.load(open(path)).items():
            prior[lang] = {"a": float(d["affine_a"]), "b": float(d["affine_b"]), "resid_p99": float(d["resid_p99"])}
    return prior


def expected_frames(prior, lang, nchar):
    p = prior.get(lang, PRIOR_DEFAULT)
    return p["a"] + p["b"] * nchar, p["resid_p99"]


def split_text(text, lang, prior, chunk_s):
    """Split at sentence terminals into pieces of about chunk_s seconds; commas, then whitespace, for run-on text."""
    if chunk_s <= 0:
        return [text]
    target = chunk_s * FPS
    pieces = [p.strip() for p in re.split(f"(?<=[{re.escape(TERMINALS)}])\\s+", text) if p.strip()]
    out, cur = [], ""
    for p in pieces:
        cand = (cur + " " + p).strip() if cur else p
        if cur and expected_frames(prior, lang, len(cand))[0] > target:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)
    final = []
    for piece in out:
        if expected_frames(prior, lang, len(piece))[0] <= target * 1.3:
            final.append(piece)
            continue
        cur = ""
        for w in re.split(r"(?<=[,،])\s+|\s+", piece):
            cand = (cur + " " + w).strip() if cur else w
            if cur and expected_frames(prior, lang, len(cand))[0] > target:
                final.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            final.append(cur)
    return final


def load_named_voices(voices_json):
    d = json.load(open(voices_json))
    items = d.get("voices", d) if isinstance(d, dict) else d
    out = []
    for it in items:
        if it.get("mode", "named") != "named":
            continue
        name = it.get("name") or it.get("display_name") or it.get("id")
        out.append(Voice(name, "named", meta={**it, "name": name}))
    return out
