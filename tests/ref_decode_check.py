"""Decode with the release's own eager reference (scripts/s2_serve_controls.ServeSession, PyTorch) and with the MLX port
under the same settings, and transcribe both with Whisper. Separates model behaviour from port bugs. Run in .venv-ref."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "maya2-native"
for p in (ROOT, VENDOR, VENDOR / "scripts"):
    sys.path.insert(0, str(p))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import MimiModel, pipeline  # noqa: E402

import s2_serve_controls as S  # noqa: E402
from maya2_mlx.model import Maya2TTS, SamplingConfig, _resolve  # noqa: E402

TEXT, LANG, VOICE, WL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SEEDS = [int(s) for s in sys.argv[5:]] or [1, 2, 3]


def main():
    from huggingface_hub import snapshot_download
    from maya3.prompt import NamedSpeakerBuilder, Voice
    from tts.data.packing_v2 import PackerV2
    from tts.model.load_s3 import load_model_for_eval
    d = _resolve("maya-research/maya2-native")
    ref, _ = load_model_for_eval(str(d), device="cpu", dtype=torch.float32,
                                 cfg={"model": {"backbone": str(d / "backbone"), "attn": "sdpa"}, "data": {"max_tokens": 8192}},
                                 config=str(VENDOR / "configs" / "s3_fresh_v1.yaml"), require_anchor=True, verbose=False)
    mimi = MimiModel.from_pretrained(snapshot_download("kyutai/mimi", allow_patterns=["config.json", "model.safetensors"])).eval()
    packer = PackerV2(ref.backbone.tokenizer, ref.backbone.token_ids, max_tokens=8192)
    v = Voice(VOICE, "named", builder=NamedSpeakerBuilder(), meta={"name": VOICE})
    v.prefix, v.spk_refs = None, np.zeros((1, 192), np.float32)
    ex = v.builder.example(packer, v, TEXT, LANG)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cpu", torch_dtype=torch.float32)

    def tr(w):
        return asr({"raw": w, "sampling_rate": 24000}, generate_kwargs={"language": WL, "task": "transcribe"})["text"].strip()

    prior = S.load_prior(str(VENDOR / "configs" / "text_frames_prior.json"))
    ef, er = S.expected_frames(prior, LANG, len(TEXT))
    mem = ref.depth.spk_memory(torch.zeros(1, 1, 192), "ecapa", null_mask=torch.tensor([True]))
    for seed in SEEDS:
        torch.manual_seed(seed)
        with torch.inference_mode():
            emb, _ = S.prompt_from_example(ex, ref, "cpu", torch.float32, 32)
            sess = S.ServeSession(ref, mem, torch.ones(1, 1, dtype=torch.bool), q_serve=32, temp=0.7, top_k=64, device="cpu")
            codes, reason, _ = sess.speak(emb, int(2 * (ef + er) + 40), stop="controlled", min_frames=max(4, int(0.5 * ef)))
            w = mimi.decode(codes[None])[0][0, 0].numpy()
        print(f"RESULT torch-ref seed={seed} {len(w) / 24000:.1f}s {reason}: {tr(w)}", flush=True)
    del ref
    tts = Maya2TTS(d, dtype=mx.float32, depth_dtype=mx.float32,
                   sampling=SamplingConfig(cb0_temp=0.7, cb0_rep=0.0))
    for seed in SEEDS:
        info = {}
        w = tts.generate(TEXT, LANG, VOICE, normalize=False, seed=seed, info=info)
        print(f"RESULT mlx      seed={seed} {info['audio_s']:.1f}s {info['stops'][-1]['reason']}: {tr(w)}", flush=True)


if __name__ == "__main__":
    main()
