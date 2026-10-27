"""Maya 2 Native (maya-research/maya2-native) in MLX, built on mlx-audio's Qwen3 and Mimi.

Architecture (third_party/maya2-native/tts/model):
  backbone      Qwen3-1.7B over text tokens and audio frames at 12.5 Hz; cb0 head = 2048 codes + <aeos>
  audio_in      backbone-side frame embedding: RMSNorm(sum_l table[l, code_l] / sqrt(32)); a learned <aeos> input
  speaker_anchor ECAPA-192 (or a learned null) -> MLP, added at every span's <|audio|> sentinel
  depth         DepthV2: levels 1..31 from the frame's hidden state; 11 weight groups x 4 layers (1024 wide), each layer
                followed by a cross-attention over ONE speaker-memory token. With one key the attention weights are 1,
                so that cross-attention is a per-voice constant o(v(mem)), computed once per voice.
  codec         Mimi, 32 codebooks, 24 kHz, decoded frame by frame (mlx-audio's streaming decoder)

Decoding follows the official engine (maya3/engine.py): cb0 at temperature 0.9 / top-k 64 with a 0.8 presence penalty
over the last 12 frames; controlled stop after a text-derived minimum (argmax == <aeos>, p(<aeos>) >= 0.35 or
cumulative hazard >= 0.6); duration cap and EOS-boost loop guard (maya3/guards.py); depth at temperature 0.7 / top-k 64.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen3 import ModelArgs as Qwen3Args
from mlx_audio.lm.models.qwen3 import Qwen3Model

from .prompt import (AEOS_CLASS, KIND_AEOS, KIND_AUDIO, LANGS, N_CODES, N_Q, Packer, Voice, expected_frames,
                     load_named_voices, load_prior, split_text)

VENDOR = Path(__file__).resolve().parents[1] / "third_party" / "maya2-native"
if not (VENDOR / "maya3" / "text").is_dir():
    raise ImportError(f"{VENDOR} is empty: the Maya 2 Native submodule is missing. Run "
                      "`git submodule update --init` in the project folder (or clone with --recursive).")
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))          # maya3.text (normaliser) and maya3.guards are pure python / numpy

REPO = "maya-research/maya2-native"
MIMI_REPO = "kyutai/moshiko-pytorch-bf16"   # the same Mimi as kyutai/mimi, in the layout mlx-audio loads
SR = 24000
FRAME = 1920
D_DEPTH, H_DEPTH, LAYERS_DEPTH = 1024, 16, 4
GROUP_OF_LEVEL = [None] + list(range(8)) + [8] * 8 + [9] * 8 + [10] * 7
N_GROUPS = 11
# audio_in is torch.nn.RMSNorm(eps=None). For bf16 inputs torch still uses the fp32 epsilon (the op computes in fp32),
# not finfo(bfloat16).eps = 2^-7. It matters: the pre-norm mean square is ~2.5e-3, and 2^-7 shrinks every audio
# embedding ~2x, which garbles speech (Whisper CER 0.20-0.52 vs 0.00-0.04 with this value; tests/quality_sweep.py).
RMS_EPS = float(np.finfo(np.float32).eps)


@dataclass
class SamplingConfig:
    cb0_temp: float = 0.9
    cb0_top_k: int = 64
    cb0_rep: float = 0.8
    cb0_rep_win: int = 12
    depth_temp: float = 0.7
    depth_top_k: int = 64
    greedy_from: Optional[int] = None   # depth levels >= this are taken by argmax (None = all sampled, the engine default)
    hard_cap: int = 1500           # frames per piece (120 s)
    chunk_s: float = 20.0          # long text is split into ~20 s pieces, each in a fresh context
    gap_s: float = 0.15            # silence between pieces
    stream_every: int = 3          # frames per streamed chunk (240 ms)


# ---------------------------------------------------------------- modules
class AudioIn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.table = nn.Embedding(N_Q * N_CODES, d)
        self.aeos_in = mx.zeros((d,))
        self.norm_weight = mx.ones((d,))
        self._offsets = mx.arange(N_Q) * N_CODES

    def _norm(self, x):
        return mx.fast.rms_norm(x, self.norm_weight, RMS_EPS)

    def frames(self, codes):
        """codes (N, 32) -> (N, d)"""
        e = self.table(codes + self._offsets).astype(mx.float32)
        return self._norm(e.sum(1) / math.sqrt(N_Q))

    def aeos(self):
        return self._norm(self.aeos_in.astype(mx.float32))


class SpeakerAnchor(nn.Module):
    def __init__(self, d, input_dim=192, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.null_vec = mx.zeros((input_dim,))

    def __call__(self, ecapa: Optional[mx.array]):
        x = self.null_vec if ecapa is None else ecapa
        return self.fc2(nn.gelu(self.fc1(x.astype(self.fc1.weight.dtype))))


class DepthBlock(nn.Module):
    """Pre-norm causal self-attention + GELU FFN (DepthV2 _SelfBlock)."""

    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D_DEPTH), nn.LayerNorm(D_DEPTH)
        self.qkv = nn.Linear(D_DEPTH, 3 * D_DEPTH, bias=False)
        self.o = nn.Linear(D_DEPTH, D_DEPTH, bias=False)
        self.f1 = nn.Linear(D_DEPTH, 3 * D_DEPTH, bias=False)
        self.f2 = nn.Linear(3 * D_DEPTH, D_DEPTH, bias=False)

    def qkv_heads(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(self.n1(x)).reshape(B, T, 3, H_DEPTH, D_DEPTH // H_DEPTH).transpose(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]            # (B, H, T, hd) each

    def finish(self, x, a):
        B, _, T, _ = a.shape
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, T, D_DEPTH))
        return x + self.f2(nn.gelu(self.f1(self.n2(x))))


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.LayerNorm(D_DEPTH)
        self.q = nn.Linear(D_DEPTH, D_DEPTH, bias=False)
        self.kv = nn.Linear(D_DEPTH, 2 * D_DEPTH, bias=False)
        self.o = nn.Linear(D_DEPTH, D_DEPTH, bias=False)

    def const(self, mem):
        """Output of the cross-attention over a single memory token: softmax over one key is 1, so it is o(v)."""
        return self.o(self.kv(mem)[..., D_DEPTH:])


class Depth(nn.Module):
    def __init__(self, d_backbone):
        super().__init__()
        self.in_proj = nn.Linear(d_backbone, D_DEPTH, bias=False)
        self.code_lo = nn.Embedding((N_Q - 1) * N_CODES, 128)
        self.code_up = nn.Linear(128, D_DEPTH, bias=False)
        self.level_emb = nn.Embedding(N_Q, D_DEPTH)
        self.groups = [[DepthBlock() for _ in range(LAYERS_DEPTH)] for _ in range(N_GROUPS)]
        self.cross = [CrossBlock() for _ in range(LAYERS_DEPTH)]
        self.norm = nn.LayerNorm(D_DEPTH)
        self.heads = [nn.Linear(D_DEPTH, N_CODES, bias=False) for _ in range(N_Q - 1)]
        self.spk_proj_ecapa = nn.Linear(192, D_DEPTH, bias=False)
        self.null_mem = mx.zeros((1, 1, D_DEPTH))

    def speaker_consts(self, ecapa: Optional[mx.array]):
        mem = self.null_mem if ecapa is None else self.spk_proj_ecapa(ecapa.astype(self.null_mem.dtype))[None, None]
        return [c.const(mem) for c in self.cross]          # per layer (1, 1, D)

    def code_embed(self, code, level):
        """Prefix position level+1: the level's code, embedded through its private rows of code_lo."""
        return self.code_up(self.code_lo(code + level * N_CODES)) + self.level_emb.weight[level + 1]

    def frame(self, h, cb0, consts, temp, top_k, greedy_from=None):
        """Levels 1..31 for one frame. h (1, d_backbone), cb0 (1,) -> codes (1, 32), all lazily on the device.

        Within a weight group the self-attention is causal and every prefix position is embedded from its own slot,
        so the bands 9-16 / 17-24 / 25-31 (groups 8 / 9 / 10) run one full pass at their first level and then extend
        a per-layer K/V cache by one position per level (serve/fast_depth.FastDepth._compute, same arithmetic)."""
        prefix = [(self.in_proj(h) + self.level_emb.weight[0])[:, None], self.code_embed(cb0, 0)[:, None]]
        codes = [cb0]
        kv = None
        previous = None
        for level in range(1, N_Q):
            group = GROUP_OF_LEVEL[level]
            incremental = group == previous
            if incremental:
                x = prefix[level]
            else:
                x = mx.concatenate(prefix[:level + 1], 1)
                kv = [None] * LAYERS_DEPTH
            T = x.shape[1]
            for li, block in enumerate(self.groups[group]):
                q, k, v = block.qkv_heads(x)
                if incremental:
                    k = mx.concatenate([kv[li][0], k], 2)
                    v = mx.concatenate([kv[li][1], v], 2)
                kv[li] = (k, v)
                a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D_DEPTH // H_DEPTH) ** -0.5,
                                                         mask="causal" if T > 1 else None)
                x = block.finish(x, a) + consts[li]
            logits = self.heads[level - 1](self.norm(x[:, -1])).astype(mx.float32)
            if greedy_from is not None and level >= greedy_from:
                code = mx.argmax(logits, -1)
            else:
                logits = logits / temp
                kth = mx.min(mx.topk(logits, top_k, axis=-1), axis=-1, keepdims=True)
                code = mx.random.categorical(mx.where(logits < kth, -mx.inf, logits))
            codes.append(code)
            if level < N_Q - 1:
                prefix.append(self.code_embed(code, level)[:, None])
            previous = group
        return mx.stack(codes, -1)


class Maya2(nn.Module):
    def __init__(self, backbone_args: Qwen3Args):
        super().__init__()
        d = backbone_args.hidden_size
        self.backbone = Qwen3Model(backbone_args)
        self.cb0_head = nn.Linear(d, N_CODES + 1, bias=False)
        self.audio_in = AudioIn(d)
        self.speaker_anchor = SpeakerAnchor(d)
        self.depth = Depth(d)

    @staticmethod
    def sanitize(weights):
        out = {}
        for k, v in weights.items():
            k = k.removeprefix("model.")
            if k.startswith("backbone.lm.model."):
                k = "backbone." + k[len("backbone.lm.model."):]
            elif k == "backbone.cb0_head.weight":
                k = "cb0_head.weight"
            elif k == "audio_in.norm.weight":
                k = "audio_in.norm_weight"
            elif k.startswith("depth.spk_proj_campp") or k.startswith("backbone.lm.lm_head"):
                continue                       # CAM++ speaker path is unused by the release; lm_head is tied
            out[k] = v
        return out


# ---------------------------------------------------------------- the TTS object
def _resolve(repo):
    from huggingface_hub import snapshot_download
    p = Path(os.path.expanduser(repo))
    if p.is_dir():
        return p
    return Path(snapshot_download(repo, allow_patterns=["*.safetensors", "*.safetensors.index.json", "maya3_config.json",
                                                        "backbone/*", "voices/*"]))


class Maya2TTS:
    """Load once, then stream(text, lang, voice) -> int16 PCM chunks (24 kHz mono)."""

    sample_rate = SR

    def __init__(self, model_dir, dtype=mx.bfloat16, depth_dtype=mx.float16, sampling: SamplingConfig = None, log=print):
        from mlx_audio.codec.models.mimi import Mimi
        from maya3 import guards as G
        t0 = time.perf_counter()
        self.G = G
        self.dir = Path(model_dir)
        self.cfg = sampling or SamplingConfig()
        bcfg = json.loads((self.dir / "backbone" / "config.json").read_text())
        bcfg["rope_theta"] = bcfg.get("rope_theta") or bcfg["rope_parameters"]["rope_theta"]
        self.model = Maya2(Qwen3Args.from_dict(bcfg))
        weights = {}
        for f in sorted(glob.glob(str(self.dir / "*.safetensors"))):
            weights.update(mx.load(f))
        self.model.load_weights(list(Maya2.sanitize(weights).items()), strict=True)
        del weights
        self.model.set_dtype(dtype)
        self.model.depth.set_dtype(depth_dtype)
        self.model.audio_in.norm_weight = self.model.audio_in.norm_weight.astype(mx.float32)
        mx.eval(self.model.parameters())
        self.model.eval()
        self.dtype = dtype
        self.packer = Packer(self.dir / "backbone")
        self.prior = load_prior(VENDOR / "configs" / "text_frames_prior.json")
        self.dur = G.DurationPrior()
        self.mimi = Mimi.from_pretrained(MIMI_REPO)
        self.mimi.eval()
        self.voices = {v.name: v for v in load_named_voices(self.dir / "voices" / "voices.json")}
        self._consts = {}
        self.load_s = time.perf_counter() - t0
        log(f"Maya 2 Native (MLX) loaded in {self.load_s:.1f} s")

    @classmethod
    def from_pretrained(cls, repo=REPO, **kw):
        return cls(_resolve(repo), **kw)

    # ------------------------------------------------------------ voices
    def _ecapa(self, voice):
        return mx.array(np.asarray(voice.ecapa, np.float32)) if voice.uses_speaker else None

    def _speaker(self, voice):
        if voice.name not in self._consts:
            e = self._ecapa(voice)
            anchor = self.model.speaker_anchor(e).astype(self.dtype)
            consts = self.model.depth.speaker_consts(e)
            mx.eval(anchor, consts)
            self._consts[voice.name] = (anchor, consts)
        return self._consts[voice.name]

    def add_voice(self, voice: Voice):
        self.voices[voice.name] = voice
        self._consts.pop(voice.name, None)

    def encode_reference(self, wav24):
        """float32 mono 24 kHz -> Mimi codes (32, F)."""
        x = mx.array(np.asarray(wav24, np.float32))[None, None]
        self.mimi.reset_state()
        codes = self.mimi.encode(x)
        mx.eval(codes)
        return np.array(codes[0]).astype(np.int64)

    # ------------------------------------------------------------ synthesis
    def _embed_prompt(self, p, anchor):
        m = self.model
        x = m.backbone.embed_tokens(mx.array(p.tokens)).astype(mx.float32)
        audio = np.nonzero(p.kind == KIND_AUDIO)[0]
        if len(audio):
            x[mx.array(audio)] = m.audio_in.frames(mx.array(p.codes[audio]))
        aeos = np.nonzero(p.kind == KIND_AEOS)[0]
        if len(aeos):
            x[mx.array(aeos)] = m.audio_in.aeos()
        x = x.astype(self.dtype)
        s = mx.array(p.sentinels)
        x[s] = x[s] + anchor
        return x[None]

    def stream(self, text, lang="en", voice="Ahana", normalize=True, seed=None, cancel=None, info=None) -> Iterator[np.ndarray]:
        if lang not in LANGS:
            raise ValueError(f"unsupported language {lang!r}; choose one of {LANGS}")
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")
        if normalize:
            from maya3.text import normalise_text
            text = normalise_text(text, lang)
        v = self.voices[voice] if voice in self.voices else \
            next(x for k, x in self.voices.items() if k.lower() == str(voice).lower())
        mx.random.seed(seed if seed is not None else int.from_bytes(os.urandom(4), "little"))
        pieces = split_text(text, lang, self.prior, self.cfg.chunk_s)
        stats = {"pieces": len(pieces), "stops": [], "frames": 0, "ttfa_s": None}
        t0 = time.perf_counter()
        for i, piece in enumerate(pieces):
            if i:
                yield np.zeros(int(self.cfg.gap_s * SR), np.int16)
            for pcm in self._piece(piece, lang, v, cancel, stats):
                if stats["ttfa_s"] is None:
                    stats["ttfa_s"] = time.perf_counter() - t0
                yield pcm
            if cancel is not None and cancel.is_set():
                break
        stats["wall_s"] = time.perf_counter() - t0
        stats["audio_s"] = stats["frames"] * FRAME / SR
        stats["rtf"] = stats["wall_s"] / max(stats["audio_s"], 1e-9)
        if info is not None:
            info.update(stats)

    def generate(self, text, lang="en", voice="Ahana", **kw):
        chunks = list(self.stream(text, lang, voice, **kw))
        return np.concatenate(chunks).astype(np.float32) / 32767.0 if chunks else np.zeros(0, np.float32)

    def _plan(self, text, lang):
        G, cfg = self.G, self.cfg
        g = G.GuardConfig()
        ef, er = expected_frames(self.prior, lang, len(text))
        loose = min(cfg.hard_cap, int(2.0 * (ef + er) + 40))
        cap = min(loose, max(int(ef * 0.5) + 8, self.dur.cap_frames(text, lang, g.dur_k, g.dur_margin_s)))
        suspend = G.suspend_windows_v2(text, lang, self.dur)
        return g, cap, loose, max(4, int(0.5 * ef)), suspend

    def _piece(self, text, lang, v, cancel, stats):
        from mlx_audio.codec.models.mimi import MimiStreamingDecoder
        m, cfg = self.model, self.cfg
        g, cap, loose, minimum, suspend = self._plan(text, lang)
        anchor, consts = self._speaker(v)
        cache = [KVCache() for _ in m.backbone.layers]
        h = m.backbone(None, cache=cache, input_embeddings=self._embed_prompt(self.packer.prompt(v, text, lang), anchor))[:, -1]
        decoder = MimiStreamingDecoder(self.mimi)
        hist = np.full(g.hist, -1, np.int64)
        recent, hazard, bias, boost_at, n_boost = [], 0.0, 0.0, None, 0
        frames, pending, reason = 0, [], "cap"
        while True:
            if cancel is not None and cancel.is_set():
                reason = "cancelled"
                break
            # ---- cb0 + stop decision: the one host sync per frame
            logits = m.cb0_head(h).astype(mx.float32)[0] / cfg.cb0_temp
            adj = np.zeros(N_CODES + 1, np.float32)
            adj[AEOS_CLASS] = bias
            if cfg.cb0_rep > 0 and recent:
                adj[sorted(set(recent))] -= cfg.cb0_rep
            logits = logits + mx.array(adj)
            kth = mx.min(mx.topk(logits, cfg.cb0_top_k), keepdims=True)
            logits = mx.where(logits < kth, -mx.inf, logits)
            probs = mx.softmax(logits)
            cont = mx.where(mx.arange(N_CODES + 1) == AEOS_CLASS, -mx.inf, logits)
            cb0 = mx.random.categorical(cont)
            pa, arg = probs[AEOS_CLASS], mx.argmax(probs)
            mx.eval(pa, arg, cb0, pending)
            if len(pending) >= cfg.stream_every:
                pcm = np.array(mx.concatenate(pending, -1)).reshape(-1)
                pending = []
                yield (np.clip(pcm, -1, 1) * 32767).round().astype(np.int16)
            pa, arg, c0 = float(pa), int(arg), int(cb0)
            if frames >= minimum:
                hazard = 1 - (1 - hazard) * (1 - pa)
                if arg == AEOS_CLASS:
                    reason = "aeos_argmax"
                    break
                if pa >= 0.35:
                    reason = "aeos_tau"
                    break
                if hazard >= 0.6:
                    reason = "aeos_hazard"
                    break
            if cfg.cb0_rep_win > 0:
                recent = (recent + [c0])[-cfg.cb0_rep_win:]
            # ---- depth, feedback, next backbone step and the frame's audio: queued lazily, evaluated with the next sync
            codes = m.depth.frame(h.astype(m.depth.in_proj.weight.dtype), cb0[None], consts, cfg.depth_temp, cfg.depth_top_k,
                                 cfg.greedy_from)
            fb = m.audio_in.frames(codes).astype(self.dtype)
            h = m.backbone(None, cache=cache, input_embeddings=fb[None])[:, -1]
            pending.append(decoder.decode_frames(codes[:, :, None])[0, 0])
            mx.async_eval(pending[-1], h)        # the GPU starts while the host runs the guards below
            frames += 1
            hist[:-1] = hist[1:]
            hist[-1] = c0
            if frames >= cap:
                reason = "cap_guard" if cap < loose else "cap"
                break
            if g.loop and frames >= g.start and frames % g.every == 0 and not any(a <= frames <= b for a, b in suspend):
                hit = self.G.loop_detect_batch(hist[None], np.array([frames]), g)[0]
                if hit is not None:
                    if boost_at is None:
                        boost_at, bias = frames, g.boost
                        n_boost += 1
                    if frames - boost_at >= g.backstop:
                        reason = "loop_backstop"
                        break
                elif boost_at is not None:
                    boost_at, bias = None, 0.0
        if pending:
            pcm = np.array(mx.concatenate(pending, -1)).reshape(-1)
            yield (np.clip(pcm, -1, 1) * 32767).round().astype(np.int16)
        stats["frames"] += frames
        stats["stops"].append({"reason": reason, "frames": frames, "cap": cap, "n_boost": n_boost})
