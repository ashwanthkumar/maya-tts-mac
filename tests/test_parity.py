"""Parity of the MLX port against the release's PyTorch reference (third_party/maya2-native), both in fp32.

    .venv/bin/python tests/test_parity.py            # needs torch + transformers (dev only; the app does not)

Checks: prompt packing token for token (named voice, same-language clone, cross-language clone), composed prompt
embeddings, backbone hidden state, cb0 logits, and one frame of greedy depth codes.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "maya2-native"
for p in (ROOT, VENDOR, VENDOR / "scripts"):
    sys.path.insert(0, str(p))

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

from maya2_mlx.model import Maya2TTS, _resolve  # noqa: E402
from maya2_mlx.prompt import Voice  # noqa: E402


def ref_example(model, packer, text, lang, ref_voice):
    import s2_serve_controls as S
    from maya3.prompt import MODES
    builder = ref_voice.builder
    ref_voice.prefix = builder.prefix(packer, ref_voice)
    ref_voice.spk_refs = (np.asarray(ref_voice.ecapa, np.float32).reshape(1, 192) if ref_voice.ecapa is not None
                          else np.zeros((1, 192), np.float32))
    ex = builder.example(packer, ref_voice, text, lang)
    with torch.inference_mode():
        emb, n = S.prompt_from_example(ex, model, "cpu", torch.float32, 32)
    return ex, emb, n


def main():
    from maya3.prompt import NamedSpeakerBuilder, V5Builder
    from maya3.prompt import Voice as RefVoice
    from tts.data.packing_v2 import PackerV2
    from tts.model.load_s3 import load_model_for_eval
    import s2_serve_controls as S

    d = _resolve("maya-research/maya2-native")
    torch.manual_seed(0)
    ref, _ = load_model_for_eval(str(d), device="cpu", dtype=torch.float32,
                                 cfg={"model": {"backbone": str(d / "backbone"), "attn": "sdpa"}, "data": {"max_tokens": 8192}},
                                 config=str(VENDOR / "configs" / "s3_fresh_v1.yaml"), require_anchor=True, verbose=False)
    rpacker = PackerV2(ref.backbone.tokenizer, ref.backbone.token_ids, max_tokens=8192)
    tts = Maya2TTS(d, dtype=mx.float32, depth_dtype=mx.float32)

    rng = np.random.default_rng(0)
    codes = rng.integers(0, 2048, (32, 40))
    ecapa = rng.standard_normal(192).astype(np.float32)
    ecapa /= np.linalg.norm(ecapa)
    cases = [
        ("named", Voice("Ahana", "named", meta={"name": "Ahana"}),
         RefVoice("Ahana", "named", builder=NamedSpeakerBuilder(), meta={"name": "Ahana"}), "வணக்கம், எப்படி இருக்கிறீர்கள்?", "ta"),
        ("clone same-lang", Voice("c1", "icl_spk", ref_text="नमस्ते दोस्तों", ref_lang="hi", ref_codes=codes, ecapa=ecapa),
         RefVoice("c1", "icl_spk", clips=[{"text": "नमस्ते दोस्तों", "codes": codes, "lang": "hi"}], ecapa=ecapa,
                  builder=V5Builder()), "आपका ऑर्डर कल पहुँच जाएगा।", "hi"),
        ("clone cross-lang", Voice("c2", "icl_spk", ref_text="नमस्ते दोस्तों", ref_lang="hi", ref_codes=codes, ecapa=ecapa),
         RefVoice("c2", "icl_spk", clips=[{"text": "नमस्ते दोस्तों", "codes": codes, "lang": "hi"}], ecapa=ecapa,
                  builder=V5Builder()), "Hello, your order ships tomorrow.", "en"),
    ]
    ok = True
    for name, v, rv, text, lang in cases:
        ex, remb, n = ref_example(ref, rpacker, text, lang, rv)
        p = tts.packer.prompt(v, text, lang)
        same = (len(p.tokens) == n and np.array_equal(p.tokens, ex["tokens"][:n]) and np.array_equal(p.kind, ex["kind"][:n])
                and np.array_equal(p.codes, ex["codes"][:n]))
        tts._consts.pop(v.name, None)
        anchor, consts = tts._speaker(v)
        memb = np.array(tts._embed_prompt(p, anchor))
        e_err = float(np.abs(memb - remb.numpy()).max())
        with torch.inference_mode():
            rh, _ = ref.backbone(remb)
            rh = rh[:, -1]
            rlog = ref.backbone.cb0_logits(rh)[0].numpy()
        from mlx_audio.lm.models.cache import KVCache
        cache = [KVCache() for _ in tts.model.backbone.layers]
        mh = tts.model.backbone(None, cache=cache, input_embeddings=mx.array(memb))[:, -1]
        mlog = np.array(tts.model.cb0_head(mh)[0])
        h_err = float(np.abs(np.array(mh) - rh.numpy()).max())
        l_err = float(np.abs(mlog - rlog).max())
        cb0 = int(np.argmax(rlog[:2048]))
        refs_t = torch.as_tensor(np.asarray(ecapa if v.uses_speaker else np.zeros(192, np.float32)).reshape(1, 1, 192))
        mem = ref.depth.spk_memory(refs_t, "ecapa", null_mask=torch.tensor([not v.uses_speaker]))
        with torch.inference_mode():
            rcodes = S.depth_generate(ref.depth, rh, torch.tensor([cb0]), mem, 32, 1.0, 1, 0.0,
                                      torch.ones(1, 1, dtype=torch.bool), greedy_from=1)[0].numpy()
        mcodes = np.array(tts.model.depth.frame(mx.array(rh.numpy()), mx.array([cb0]), consts, 1.0, 1))[0]
        match = int((rcodes == mcodes).sum())
        print(f"{name:17s} prompt={'OK' if same else 'DIFF'} T={n}  emb max|d|={e_err:.2e}  hidden max|d|={h_err:.2e}  "
              f"cb0 logits max|d|={l_err:.2e}  greedy depth codes {match}/32")
        ok &= same and e_err < 1e-3 and l_err < 5e-2 and match >= 30
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
