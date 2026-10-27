"""Decoding-settings sweep scored by Whisper character error rate (dev check; run in .venv-ref).

    .venv-ref/bin/python tests/quality_sweep.py [n_seeds]

CER against the input text is a relative measure here (Whisper's own Tamil/Hindi errors are in every row); compare
configs, not absolute numbers. Sentences have no digits so normalisation cannot change the reference.
"""
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from maya2_mlx.model import Maya2TTS, SamplingConfig  # noqa: E402

SENTENCES = [
    ("en", "Siddhant", "english", "Your order has shipped and will arrive tomorrow. Please keep your phone nearby for the delivery call."),
    ("hi", "Ahana", "hindi", "नमस्ते, आपका ऑर्डर कल पहुँच जाएगा। कृपया डिलीवरी के समय अपना फ़ोन पास रखें।"),
    ("ta", "Anirban", "tamil", "வணக்கம்! இன்று உங்கள் நாள் எப்படி இருக்கிறது? நான் உங்களுக்கு எப்படி உதவ முடியும்?"),
]
CONFIGS = {
    "engine default": SamplingConfig(),
    "greedy>=9": SamplingConfig(greedy_from=9),
    "cb0 0.7 + greedy>=9": SamplingConfig(cb0_temp=0.7, greedy_from=9),
    "greedy depth": SamplingConfig(greedy_from=1),
}


def norm(t):
    t = unicodedata.normalize("NFC", t.lower())
    return re.sub(r"[\s\W_]+", "", t)


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return min(1.0, d[len(h)] / max(1, len(r)))


if __name__ == "__main__":
    seeds = range(1, 1 + (int(sys.argv[1]) if len(sys.argv) > 1 else 5))
    tts = Maya2TTS.from_pretrained(log=lambda m: None)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device="cpu",
                   torch_dtype=torch.float32)
    table = {}
    for name, cfg in CONFIGS.items():
        tts.cfg = cfg
        for lang, voice, wl, text in SENTENCES:
            scores = []
            for seed in seeds:
                info = {}
                w = tts.generate(text, lang, voice, normalize=False, seed=seed, info=info)
                hyp = asr({"raw": w, "sampling_rate": 24000}, generate_kwargs={"language": wl, "task": "transcribe"})["text"]
                scores.append(cer(text, hyp[: 3 * len(text)]))
                print(f"RESULT {name:22s} {lang} seed={seed} cer={scores[-1]:.2f} {info['stops'][-1]['reason']}: {hyp[:120]}",
                      flush=True)
            table[(name, lang)] = scores
    print("SUMMARY  mean CER (fraction of takes with CER < 0.1)")
    for name in CONFIGS:
        cells = [f"{lang}: {np.mean(table[(name, lang)]):.3f} ({np.mean(np.array(table[(name, lang)]) < 0.1):.0%})"
                 for lang, *_ in SENTENCES]
        print(f"SUMMARY  {name:22s} " + "   ".join(cells), flush=True)
